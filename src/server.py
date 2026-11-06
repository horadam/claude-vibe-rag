"""FastAPI application exposing the RAG pipeline over HTTP."""

import json
import logging
import time
from pathlib import Path

import anthropic
import openai
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import CORPUS_DIR, EVAL_RUNS_DIR, TOP_K
from src.evaluation.report import list_runs, load_run
from src.evaluation.scoring import is_refusal
from src.generate import REFUSAL_MESSAGE, GenerationResult, generate
from src.retrieve import get_chunk, index_size, retrieve

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Northwind Robotics HR Assistant")
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Provider errors ─────────────────────────────────────────────────────────
# Upstream API failures reach the browser with the provider's own message,
# as 502 Bad Gateway: the request was fine, a service behind us failed.


@app.exception_handler(openai.APIError)
def openai_error(request: Request, exc: openai.APIError) -> JSONResponse:
    """Surface OpenAI (embedding) failures with their original message."""
    logger.error("OpenAI API error: %s", exc.message)
    return JSONResponse(status_code=502, content={"detail": f"OpenAI API error: {exc.message}"})


@app.exception_handler(anthropic.APIError)
def anthropic_error(request: Request, exc: anthropic.APIError) -> JSONResponse:
    """Surface Anthropic (generation) failures with their original message."""
    logger.error("Anthropic API error: %s", exc.message)
    return JSONResponse(status_code=502, content={"detail": f"Anthropic API error: {exc.message}"})


# ── Pages ───────────────────────────────────────────────────────────────────


def _corpus_files() -> set[str]:
    """Filenames that may be served as handbook pages."""
    return {p.name for p in CORPUS_DIR.glob("*.md")}


def _doc_title(path: Path) -> str:
    """First H1 of a handbook file, or its filename if it has none."""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return path.name


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    """Serve the chat UI."""
    return HTMLResponse(content=open("static/index.html").read())


@app.get("/evals", response_class=HTMLResponse)
def evals_page() -> HTMLResponse:
    """Serve the eval results page."""
    return HTMLResponse(content=open("static/evals.html").read())


@app.get("/docs/{filename}", response_class=HTMLResponse)
def doc(filename: str) -> HTMLResponse:
    """Render a corpus markdown file as a styled HTML page."""
    if filename not in _corpus_files():
        raise HTTPException(status_code=404, detail=f"{filename} not found")
    markdown_content = (CORPUS_DIR / filename).read_text(encoding="utf-8")
    documents = [
        {"filename": name, "title": _doc_title(CORPUS_DIR / name)}
        for name in sorted(_corpus_files())
    ]
    documents.sort(key=lambda d: d["title"])
    # Embedded as JSON inside <script type="application/json">. Escaping "<"
    # means no document text can close the script tag and inject markup.
    payload = json.dumps(
        {"filename": filename, "markdown": markdown_content, "documents": documents}
    )
    payload = payload.replace("<", "\\u003c")
    return HTMLResponse(content=open("static/doc.html").read().replace("__DOC_JSON__", payload))


# ── API ─────────────────────────────────────────────────────────────────────


class AskRequest(BaseModel):
    question: str


class CitationOut(BaseModel):
    source: str
    chunk_preview: str


class RetrievedOut(BaseModel):
    """One retrieved chunk, as shown in the answer inspector."""

    source: str
    title: str
    chunk_index: int
    distance: float
    text: str
    cited: bool


class TimingOut(BaseModel):
    retrieve_ms: float
    generate_ms: float


class UsageOut(BaseModel):
    model: str
    input_tokens: int
    output_tokens: int


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    refused: bool
    top_k: int
    retrieved: list[RetrievedOut]
    timing: TimingOut
    usage: UsageOut


class HealthResponse(BaseModel):
    status: str
    doc_count: int


class PassageResponse(BaseModel):
    source: str
    chunk_index: int
    text: str


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return service status and number of indexed chunks."""
    return HealthResponse(status="ok", doc_count=index_size())


@app.get("/api/passages/{source}/{chunk_index}", response_model=PassageResponse)
def passage(source: str, chunk_index: int) -> PassageResponse:
    """Return the stored text of one chunk, for highlighting on the handbook page."""
    if source not in _corpus_files():
        raise HTTPException(status_code=404, detail=f"{source} not found")
    try:
        text = get_chunk(source, chunk_index)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if text is None:
        raise HTTPException(status_code=404, detail=f"{source} has no chunk {chunk_index}")
    return PassageResponse(source=source, chunk_index=chunk_index, text=text)


@app.get("/api/evals/runs")
def eval_runs() -> list[dict]:
    """List saved eval runs, newest first, with their headline summary."""
    return list_runs(EVAL_RUNS_DIR)


@app.get("/api/evals/runs/{run_id}")
def eval_run(run_id: str) -> dict:
    """Return one saved eval run with per-case detail."""
    record = load_run(EVAL_RUNS_DIR, run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"run {run_id} not found")
    return record


@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest) -> AskResponse:
    """Retrieve relevant chunks and generate a grounded answer."""
    if not body.question.strip():
        raise HTTPException(status_code=422, detail="question must not be empty")

    logger.info("POST /ask question=%r", body.question[:120])

    try:
        start = time.perf_counter()
        chunks = retrieve(body.question)
        retrieve_ms = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        result: GenerationResult = generate(body.question, chunks)
        generate_ms = (time.perf_counter() - start) * 1000
    except RuntimeError as exc:
        # Missing index or missing API key: the service isn't ready.
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    cited = {c.source for c in result.citations}
    logger.info(
        "POST /ask completed in %.2fs, citations=%s",
        (retrieve_ms + generate_ms) / 1000,
        sorted(cited),
    )

    return AskResponse(
        answer=result.answer,
        citations=[
            CitationOut(source=c.source, chunk_preview=c.chunk_preview)
            for c in result.citations
        ],
        refused=is_refusal(result.answer, REFUSAL_MESSAGE),
        top_k=TOP_K,
        retrieved=[
            RetrievedOut(
                source=c.source,
                title=c.title,
                chunk_index=c.chunk_index,
                distance=round(c.distance, 4),
                text=c.text,
                cited=c.source in cited,
            )
            for c in chunks
        ],
        timing=TimingOut(retrieve_ms=round(retrieve_ms, 1), generate_ms=round(generate_ms, 1)),
        usage=UsageOut(
            model=result.model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        ),
    )
