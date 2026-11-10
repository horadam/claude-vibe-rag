import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.retrieve import IndexedChunk, load_index, search, squared_l2


def chunk(doc_id: str, embedding: list[float]) -> IndexedChunk:
    source, index = doc_id.split("__")
    return IndexedChunk(doc_id, f"text {doc_id}", f"{source}.md", int(index), source.title(), embedding)


class SearchTest(unittest.TestCase):
    def test_ranks_by_distance_and_cuts_to_top_k(self) -> None:
        chunks = [chunk("far__0", [0.0, 1.0]), chunk("near__0", [1.0, 0.0]), chunk("mid__0", [0.6, 0.8])]
        results = search([1.0, 0.0], chunks, top_k=2)
        self.assertEqual([r.doc_id for r in results], ["near__0", "mid__0"])
        self.assertAlmostEqual(results[0].distance, 0.0)
        self.assertEqual(results[1].source, "mid.md")

    def test_squared_l2_of_unit_vectors_is_twice_cosine_distance(self) -> None:
        # cos(a, b) = 0.6, so 2 × (1 − 0.6) = 0.8
        self.assertAlmostEqual(squared_l2([1.0, 0.0], [0.6, 0.8]), 0.8)

    def test_mismatched_dimensions_raise(self) -> None:
        with self.assertRaises(ValueError):
            squared_l2([1.0, 0.0], [1.0])


class LoadIndexTest(unittest.TestCase):
    def test_missing_index_raises_a_clear_error(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "run ingest_cli.py"):
            load_index(Path("does/not/exist.json"))

    def test_reads_exported_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.json"
            item = {"doc_id": "a__0", "text": "t", "source": "a.md", "chunk_index": 0, "title": "A", "embedding": [1.0]}
            path.write_text(json.dumps({"embedding_model": "m", "chunks": [item]}))
            [loaded] = load_index(path)
        self.assertEqual(loaded.source, "a.md")
        self.assertEqual(loaded.embedding, [1.0])


class ServerImportTest(unittest.TestCase):
    def test_server_does_not_import_chromadb(self) -> None:
        # The deployed server has no Chroma installed; importing it must not need it.
        code = "import sys, src.server; print('chromadb' in sys.modules)"
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.strip(), "False")


if __name__ == "__main__":
    unittest.main()
