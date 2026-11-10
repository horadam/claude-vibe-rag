# RAG over Fictional HR Handbook — Spec

## Goal

A minimal but production-shaped RAG service that answers questions about a
fictional company's HR policies. Optimized for learning the full RAG loop end
to end — ingest, retrieve, generate, evaluate — not for scale.

## Stack

- Python 3.11+
- `uv` for dependency and venv management
- FastAPI + Uvicorn for the HTTP layer
- OpenAI `text-embedding-3-small` for embeddings
- ChromaDB (local persistent client at `./chroma_db`) to build the index; queries
  search its JSON export (`index/hr_handbook.json`), so the server runs without Chroma
- Anthropic `claude-sonnet-4-5` for answer generation
- `python-dotenv` for secrets

No LangChain, no LlamaIndex. The whole point is to see the moving parts.

## Project layout

```
.
├── SPEC.md
├── CLAUDE.md
├── .env.example
├── pyproject.toml
├── README.md
├── corpus/                    # ~10 fictional HR markdown docs
├── evals/
│   └── questions.json         # eval set (see Eval section)
├── src/
│   ├── __init__.py
│   ├── config.py              # env loading, model names, constants
│   ├── ingest.py              # chunking + embedding + chroma upsert
│   ├── retrieve.py            # query embedding + search over the JSON index
│   ├── generate.py            # prompt assembly + Claude call
│   └── server.py              # FastAPI app
├── ingest_cli.py              # `python ingest_cli.py` runs ingestion
├── eval_cli.py                # `python eval_cli.py` runs the eval harness
├── index/hr_handbook.json     # committed: chunks + embeddings, exported at ingest
└── chroma_db/                 # gitignored
```

## Ingestion

- Read every `*.md` file under `corpus/`.
- Chunk by paragraph, then merge until ~500 tokens (use `tiktoken` `cl100k_base`
  for counting), with ~50 token overlap between adjacent chunks.
- Each chunk gets metadata: `source` (filename), `chunk_index`, `title` (first
  H1 of the source doc).
- Embed in batches of 100 via OpenAI.
- Upsert into a single Chroma collection named `hr_handbook`.
- Print per-file chunk counts and total chunks/tokens at the end.
- Re-running ingestion should be idempotent — clear the collection first.
- Export every chunk with its embedding to `index/hr_handbook.json`.

## Retrieval

- Embed the user question with the same model used at ingest.
- Rank every chunk in `index/hr_handbook.json` by squared L2 distance (Chroma's
  default measure) and take the top-k=5. The corpus is small enough for an exact
  search in plain Python.
- Return chunks with their distances and metadata.

## Generation

- System prompt requires the model to answer **only** from provided context,
  cite sources by filename, and explicitly say "I don't have that information
  in the handbook" when the answer isn't grounded.
- User message is structured:

  ```
  CONTEXT:
  [source: pto_policy.md]
  <chunk text>

  [source: remote_work.md]
  <chunk text>

  QUESTION: <user question>
  ```

- Return `{ answer: str, citations: [{source, chunk_preview}] }`.

## API

- `POST /ask` with body `{"question": "..."}` → `{answer, citations}`.
- `GET /health` → `{"status": "ok", "doc_count": <int>}`.

## Eval harness

- `evals/questions.json` shape:

  ```json
  [
    {
      "id": "pto_accrual_basic",
      "question": "How much PTO do employees accrue per month?",
      "must_contain": ["1.5", "days"],
      "must_not_contain": [],
      "category": "single_doc"
    },
    {
      "id": "out_of_scope",
      "question": "What is the company's stock ticker symbol?",
      "must_contain": ["don't have", "handbook"],
      "must_not_contain": [],
      "category": "out_of_scope"
    }
  ]
  ```

- `eval_cli.py` runs each question through `/ask` (or directly through the
  generate pipeline — your call), checks string containment case-insensitively,
  and prints:
  - per-question PASS/FAIL with the actual answer on failures
  - overall pass rate
  - breakdown by category

- Categories must include:
  - `single_doc` — answer in one document
  - `multi_doc` — answer requires combining two documents
  - `ambiguous` — term appears in multiple docs, retrieval ranking matters
  - `out_of_scope` — answer is not in corpus, system should refuse

## Corpus generation

Generate ~10 fictional HR markdown files for a made-up company called
**Northwind Robotics**. Required topics:

1. PTO and vacation policy
2. Remote and hybrid work policy
3. Health and dental benefits
4. Parental leave
5. Expense reimbursement
6. Code of conduct
7. Performance review process
8. Compensation and equity
9. Information security policy
10. Termination and offboarding

Deliberately include:
- At least one cross-reference between docs (e.g., remote work policy referencing
  the security policy)
- Two docs that both mention "leave" with different meanings (parental vs PTO)
  — this powers an `ambiguous` eval case
- One policy detail that requires reading two docs to fully answer (e.g., total
  parental leave = base policy + PTO top-up rules)

## Out of scope

- No auth, no rate limiting, no streaming responses.
- No frontend. `curl` is fine.
- No Docker, no deploy.
- No reranking, no hybrid search.

## Definition of done

1. `uv run python ingest_cli.py` ingests the corpus cleanly.
2. `uvicorn src.server:app --reload` boots the API.
3. `curl -X POST localhost:8000/ask -H 'content-type: application/json' -d '{"question":"..."}'` returns a grounded answer with citations.
4. `uv run python eval_cli.py` runs the eval set and prints a pass-rate summary.
5. README documents setup and the four commands above.
