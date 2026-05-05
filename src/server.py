"""FastAPI application exposing the RAG pipeline over HTTP."""

import logging
import time

import chromadb
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import CHROMA_DIR, COLLECTION_NAME, CORPUS_DIR
from src.generate import GenerationResult, generate
from src.retrieve import retrieve

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Northwind Robotics HR Assistant")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    """Serve the chat UI."""
    return HTMLResponse(content=open("static/index.html").read())


@app.get("/docs/{filename}", response_class=HTMLResponse)
def doc(filename: str) -> HTMLResponse:
    """Render a corpus markdown file as a styled HTML page."""
    path = CORPUS_DIR / filename
    if not path.exists() or path.suffix != ".md":
        raise HTTPException(status_code=404, detail=f"{filename} not found")
    markdown_content = path.read_text(encoding="utf-8")
    escaped = markdown_content.replace("`", "\\`").replace("$", "\\$")
    return HTMLResponse(content=open("static/doc.html").read().replace("__MARKDOWN__", escaped))


class AskRequest(BaseModel):
    question: str


class CitationOut(BaseModel):
    source: str
    chunk_preview: str


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationOut]


class HealthResponse(BaseModel):
    status: str
    doc_count: int


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return service status and number of indexed chunks."""
    chroma = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        collection = chroma.get_collection(COLLECTION_NAME)
        doc_count = collection.count()
    except Exception:
        doc_count = 0
    return HealthResponse(status="ok", doc_count=doc_count)


@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest) -> AskResponse:
    """Retrieve relevant chunks and generate a grounded answer."""
    if not body.question.strip():
        raise HTTPException(status_code=422, detail="question must not be empty")

    logger.info("POST /ask question=%r", body.question[:120])
    t0 = time.monotonic()

    try:
        chunks = retrieve(body.question)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    result: GenerationResult = generate(body.question, chunks)

    elapsed = time.monotonic() - t0
    logger.info("POST /ask completed in %.2fs, citations=%s", elapsed, [c.source for c in result.citations])

    return AskResponse(
        answer=result.answer,
        citations=[
            CitationOut(source=c.source, chunk_preview=c.chunk_preview)
            for c in result.citations
        ],
    )
