from __future__ import annotations

import os

from dotenv import load_dotenv


load_dotenv()

TICKET_API_WSDL_URL = os.getenv("TICKET_API_WSDL_URL")
TICKET_API_BASE_URL = os.getenv("TICKET_API_BASE_URL")
TICKET_API_USERNAME = os.getenv("TICKET_API_USERNAME")
TICKET_API_PASSWORD = os.getenv("TICKET_API_PASSWORD")

KNOWLEDGE_BASE_PATH = os.getenv("KNOWLEDGE_BASE_PATH", "knowledge_base")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "railway_knowledge")


TASK3_EXPERT_DB_PATH = os.getenv("TASK3_EXPERT_DB_PATH", "task3_expert.db")
TASK3_KB_PATH = os.getenv("TASK3_KB_PATH", "knowledge_base_task3")
TASK3_CHROMA_COLLECTION_NAME = os.getenv("TASK3_CHROMA_COLLECTION_NAME", "swr_contingency_expert_kb")
TASK3_DOWNLOADS_PATH = os.getenv("TASK3_DOWNLOADS_PATH", r"C:\Users\rajvi\Downloads")
