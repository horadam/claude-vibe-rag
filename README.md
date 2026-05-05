# Northwind Robotics HR Assistant

A minimal RAG service that answers questions about a fictional company's HR
policies. Covers the full loop end to end: ingest → retrieve → generate →
evaluate — plus a polished chat UI.

**Stack:** FastAPI · ChromaDB · OpenAI embeddings · Claude (Anthropic) · Vanilla JS

---

### Demo

https://github.com/user-attachments/assets/f429311e-7f50-4b9b-ade0-ab9aa5efac46

---

## Setup

**Prerequisites:** Python 3.11+, [uv](https://docs.astral.sh/uv/)

```bash
git clone <repo>
cd claude_vibe_rag
uv sync
cp .env.example .env
```

Edit `.env` and add your API keys:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

---

## Usage

### 1. Ingest the corpus

Chunks, embeds, and upserts the HR handbook into ChromaDB. Safe to re-run —
clears the collection first.

```bash
uv run python ingest_cli.py
```

### 2. Start the server

```bash
uvicorn src.server:app --reload
```

Then open **http://localhost:8000** in your browser to use the chat UI.

### 3. Ask via curl

```bash
curl -s -X POST localhost:8000/ask \
  -H 'content-type: application/json' \
  -d '{"question": "How much PTO do employees accrue per month?"}' \
  | python3 -m json.tool
```

Check how many chunks are indexed:

```bash
curl -s localhost:8000/health | python3 -m json.tool
```

### 4. Run the eval harness

Keyword checks only (fast, free):

```bash
uv run python eval_cli.py
```

Keyword + model-graded (Claude judges each answer against a rubric):

```bash
uv run python eval_cli.py --model-grade
```

---

## Project layout

```
corpus/          10 fictional Northwind Robotics HR policy docs
evals/           Eval question set with keyword and rubric checks
static/
  index.html     Chat UI
  doc.html       Rendered handbook page (linked from citations)
  marked.min.js  Markdown renderer (vendored)
  purify.min.js  HTML sanitiser (vendored)
src/
  config.py      Constants and env loading
  ingest.py      Chunking, embedding, Chroma upsert
  retrieve.py    Query embedding + vector search
  generate.py    Prompt assembly + Claude call
  server.py      FastAPI app
ingest_cli.py    CLI entry point for ingestion
eval_cli.py      Eval harness
chroma_db/       Local vector store (gitignored, created on first ingest)
```

---

## How it works

1. **Ingest** — Each `.md` file in `corpus/` is split into ~500-token chunks
   (paragraph-boundary, with overlap). Chunks are embedded via OpenAI
   `text-embedding-3-small` and stored in a local ChromaDB collection.

2. **Retrieve** — The user's question is embedded with the same model. ChromaDB
   returns the 5 nearest chunks by cosine distance.

3. **Generate** — The chunks are assembled into a `CONTEXT` block and sent to
   `claude-sonnet-4-6` with a system prompt that requires grounded answers and
   source citations.

4. **UI** — A single-page chat interface served by FastAPI. Answers render as
   markdown. Citation pills link to fully rendered handbook pages at
   `GET /docs/{filename}`.

5. **Evaluate** — `eval_cli.py` sends each question to `/ask` and checks the
   answer against keyword assertions. With `--model-grade`, a second Claude call
   judges each answer against a human-written rubric.
