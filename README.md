# Northwind Robotics HR Assistant

A minimal RAG service that answers questions about a fictional company's HR
policies. Covers the full loop end to end: ingest → retrieve → generate →
evaluate — plus a polished chat UI.

**Stack:** FastAPI · ChromaDB · OpenAI embeddings · Claude or GPT for answers · Vanilla JS

---

### Demo

https://github.com/user-attachments/assets/f429311e-7f50-4b9b-ade0-ab9aa5efac46

---

## Setup

**Prerequisites:** Python 3.11–3.14 (pinned to 3.13 in `.python-version`; chromadb's onnxruntime has no 3.15 wheels yet), [uv](https://docs.astral.sh/uv/)

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

**Choosing a provider.** Search always embeds with OpenAI, so an OpenAI key is
always needed. Answers and the eval judge use Anthropic by default; set
`LLM_PROVIDER=openai` to run everything on OpenAI with a single key. Model
names and `JUDGE_PROVIDER` can be overridden too; see `.env.example`.

In the chat UI, **Model settings** (top right) switches between Anthropic and
OpenAI per tab and accepts your own keys. A key typed there is optional when
the server's `.env` already has one. It's kept in the tab's session storage
and sent with each question as a header. The server uses it for that request
only and never saves or logs it.

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

The harness runs every question in `evals/questions.json` through the pipeline
in-process, so the server doesn't need to be running.

```bash
uv run python eval_cli.py                    # retrieval + answer checks
uv run python eval_cli.py --judge            # add model-graded correctness and groundedness
uv run python eval_cli.py --retrieval-only   # retrieval metrics only (embedding calls only)
uv run python eval_cli.py --k 3 --label k3 --compare latest   # try a change, diff against last run
uv run python eval_cli.py --ids pip_duration,out_of_scope_ticker
```

Each run is saved to `evals/runs/<run-id>.json` (gitignored) with its config,
git commit and per-case detail. A full, unfiltered run also rewrites
[`evals/results.md`](evals/results.md). The exit code is non-zero when any case
fails, so it can gate CI.

What gets measured, and why, is explained in
[docs/FEATURES.md](docs/FEATURES.md).

### 5. Run the unit tests

Offline, no API keys needed:

```bash
uv run python -m unittest
```

---

## Project layout

```
corpus/          10 fictional Northwind Robotics HR policy docs
evals/           Eval question set with keyword and rubric checks
static/
  index.html     Chat UI with the answer inspector
  doc.html       Rendered handbook page; highlights a cited passage
  evals.html     Eval results: metrics, run comparison, per-case detail
  app.css        Shared light/dark theme tokens and header
  marked.min.js  Markdown renderer (vendored)
  purify.min.js  HTML sanitiser (vendored)
src/
  config.py      Constants and env loading
  ingest.py      Chunking, embedding, Chroma upsert
  retrieve.py    Query embedding + vector search
  generate.py    Prompt assembly + answer generation
  llm.py         Anthropic / OpenAI adapter (the only provider-specific code)
  server.py      FastAPI app
  evaluation/    Eval harness: dataset, scoring, judge, runner, report
tests/           Offline unit tests
docs/FEATURES.md How each feature works, in plain language
ingest_cli.py    CLI entry point for ingestion
eval_cli.py      Eval harness CLI
chroma_db/       Local vector store (gitignored, created on first ingest)
```

---

## How it works

1. **Ingest** — Each `.md` file in `corpus/` is split into ~500-token chunks
   (paragraph-boundary, with overlap). Chunks are embedded via OpenAI
   `text-embedding-3-small` and stored in a local ChromaDB collection.

2. **Retrieve** — The user's question is embedded with the same model. ChromaDB
   returns the 5 nearest chunks by cosine distance.

3. **Generate** — The chunks are assembled into a `CONTEXT` block and sent,
   with a system prompt that requires grounded answers and source citations,
   to `claude-sonnet-4-6` or (with `LLM_PROVIDER=openai`, or chosen in the UI)
   `gpt-5-mini`.

4. **UI** — A single-page chat served by FastAPI. Answers render as markdown,
   and each cited filename becomes a link to the exact passage on a rendered
   handbook page (`/docs/{file}?chunk=N`, highlighted). Under every answer,
   "How this answer was found" shows the retrieved chunks with distances,
   which were cited, latency and token use. `/evals` shows saved eval runs
   side by side. Provider errors are shown with their original message.

5. **Evaluate** — `eval_cli.py` runs each labelled question through the
   pipeline and scores retrieval (hit rate, recall, MRR, context precision
   against `expected_sources`) separately from the answer (keywords, refusal
   behaviour, and optionally a judge model grading correctness and
   groundedness). Results, latency and token use are saved per run.
