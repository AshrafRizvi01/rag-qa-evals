import json
import os

import pytest
from app.rag import retrieve

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "golden.json")
with open(GOLDEN_PATH, "r") as f:
    goldens = json.load(f)

@pytest.mark.parametrize("row", goldens, ids=lambda row: row["question"])
def test_expected_source_in_top_k(indexed, row):
    hits = retrieve(row["question"], indexed, k=3)
    sources = [hit["source"] for hit in hits]
    assert row["source"] in sources, f"expected {row['source']!r} in top 3, got {sources}"
