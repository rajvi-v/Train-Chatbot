from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from app.config import CHROMA_COLLECTION_NAME, CHROMA_DB_PATH, KNOWLEDGE_BASE_PATH


SUPPORTED_EXTENSIONS = {".txt", ".md"}
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 3
MAX_DISTANCE = 1.4


def load_text_documents(kb_path: str | Path = KNOWLEDGE_BASE_PATH) -> list[dict[str, str]]:
    base_path = Path(kb_path)
    if not base_path.exists():
        return []

    documents: list[dict[str, str]] = []
    for path in sorted(base_path.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        text = path.read_text(encoding="utf-8").strip()
        if text:
            documents.append({"source": str(path), "text": text})

    return documents


def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    clean_text = " ".join(text.split())
    if not clean_text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(clean_text):
        end = min(start + chunk_size, len(clean_text))
        chunks.append(clean_text[start:end].strip())
        if end == len(clean_text):
            break
        start = max(end - overlap, start + 1)

    return chunks


def ingest_documents(
    kb_path: str | Path = KNOWLEDGE_BASE_PATH,
    db_path: str | Path = CHROMA_DB_PATH,
    collection_name: str = CHROMA_COLLECTION_NAME,
) -> int:
    client = _get_chroma_client(db_path)
    collection = client.get_or_create_collection(name=collection_name)

    documents = load_text_documents(kb_path)
    if not documents:
        return 0

    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict[str, Any]] = []

    for document in documents:
        source = document["source"]
        for index, chunk in enumerate(split_text(document["text"])):
            ids.append(f"{source}:{index}")
            texts.append(chunk)
            metadatas.append({"source": source, "chunk": index})

    if not texts:
        return 0

    collection.upsert(ids=ids, documents=texts, metadatas=metadatas)
    return len(texts)


def search_knowledge_base(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    db_path: str | Path = CHROMA_DB_PATH,
    collection_name: str = CHROMA_COLLECTION_NAME,
) -> list[dict[str, Any]]:
    if not query.strip():
        return []

    try:
        client = _get_chroma_client(db_path)
        collection = client.get_collection(name=collection_name)
        result = collection.query(query_texts=[query], n_results=top_k)
    except Exception:
        return []

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    matches: list[dict[str, Any]] = []
    for text, metadata, distance in zip(documents, metadatas, distances):
        if distance is not None and float(distance) > MAX_DISTANCE:
            continue
        matches.append(
            {
                "text": text,
                "source": metadata.get("source", "knowledge base") if metadata else "knowledge base",
                "distance": distance,
            }
        )

    return matches


def format_kb_context(matches: list[dict[str, Any]], max_chars: int = 500) -> str:
    if not matches:
        return ""

    first_match = matches[0]
    text = str(first_match["text"]).strip()
    if len(text) > max_chars:
        text = f"{text[:max_chars].rstrip()}..."

    source = Path(str(first_match["source"])).name
    return f"Knowledge base note: {text}\nSource: {source}"


def answer_from_knowledge_base(matches: list[dict[str, Any]]) -> str:
    note = format_kb_context(matches)
    if note:
        return note
    return (
        "Sorry, I can help with cheap UK train tickets, train delay predictions, "
        "or questions covered by the local knowledge base."
    )


def _get_chroma_client(db_path: str | Path) -> Any:
    import chromadb

    return chromadb.PersistentClient(path=str(db_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage the local ChromaDB knowledge base.")
    parser.add_argument("command", choices=["ingest"], help="Knowledge base command to run")
    args = parser.parse_args()

    if args.command == "ingest":
        count = ingest_documents()
        print(f"Ingested {count} knowledge base chunks into {CHROMA_COLLECTION_NAME}.")


if __name__ == "__main__":
    main()
