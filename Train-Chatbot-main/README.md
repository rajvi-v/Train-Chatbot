# Railway Chatbot Base

FastAPI base chatbot for the CMP-7059B railway coursework Task 1: finding the cheapest available UK train ticket for a user's intended journey.

## Run

```powershell
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

## API

`POST /api/chat`

```json
{
  "session_id": "demo-session",
  "message": "I need the cheapest ticket from Norwich to London tomorrow"
}
```

The base version uses deterministic ticket intent and slot extraction, SQLite session/log storage, and a replaceable ticket service stub.
