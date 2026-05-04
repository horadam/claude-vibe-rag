# Project conventions for Claude Code

## Read this every session before writing code.

## Ground rules

- Read `SPEC.md` before proposing any structural changes. If something in SPEC
  is ambiguous, ask once — don't guess and proceed.
- Build in stages. Do not write the whole project in one pass. Stop after each
  stage and let the human review the diff and run the code.
- Never silently swallow errors. No bare `except:` and no `except Exception: pass`.
  If you catch, log the exception and re-raise unless there's a specific
  documented reason not to.
- Type hints on every function signature. Return types too.
- No print-debugging left in the code. Use `print()` only where SPEC explicitly
  asks for user-facing output (ingest progress, eval output).

## Tooling

- Use `uv` for everything. `uv add <pkg>` to install, `uv run <cmd>` to execute.
  Don't `pip install`. Don't create a `requirements.txt` by hand — let `uv`
  manage `pyproject.toml`.
- Python 3.11+ syntax is fine. Use `|` unions, not `Union[]`.
- `.env` is gitignored. `.env.example` is committed. Never write real keys
  anywhere.

## Dependencies — keep the list short

Allowed without asking:
- `fastapi`, `uvicorn`
- `openai`, `anthropic`
- `chromadb`
- `tiktoken`
- `python-dotenv`
- `pydantic` (comes with FastAPI anyway)
- `httpx` (for eval CLI if it hits the API over HTTP)

Anything else: ask first. Especially do not add `langchain`, `llama-index`,
`langgraph`, or any other heavyweight orchestration framework. The point of
this project is to see the moving parts.

## Code style

- Module-level constants in `SCREAMING_SNAKE_CASE` at the top of `config.py`.
- One responsibility per module. `ingest.py` doesn't know about FastAPI.
  `server.py` doesn't know about Chroma directly — it goes through `retrieve.py`.
- Functions over classes unless state genuinely needs encapsulation. The
  Chroma client and OpenAI client can be module-level singletons.
- Docstrings on public functions, one-line minimum.

## Prompts

- The system prompt for generation lives in `src/generate.py` as a module
  constant `SYSTEM_PROMPT`. It is a deliberate design surface — we will iterate
  on it during eval. Don't hide it inside an f-string in the middle of a
  function.
- When you change the system prompt, note what you changed and why in your
  message to the human.

## Errors and edge cases

- Empty corpus → ingest should warn and exit with non-zero status, not crash.
- Chroma collection missing on `/ask` → return 503 with a clear message, not 500.
- OpenAI/Anthropic API errors → propagate with the original error message
  visible. Don't wrap in a generic "something went wrong."

## What "done" means for each stage

After each stage, the code must run. Not "should run" — actually run. If you
can't test it (e.g., needs an API key the human hasn't added yet), say so
explicitly and stop. Don't pile a second stage on top of unverified first stage
work.

## When you're unsure

Ask. One specific question is better than three paragraphs of hedging or, worse,
guessing wrong and writing 200 lines that have to be thrown away.
