import pytest
from app.rag import load_docs, chunk, index
@pytest.fixture(scope="session")
def indexed():
    docs = load_docs("docs")
    chunks = []
    for doc in docs:
        chunks.extend(chunk(doc["text"], doc["source"]))
    return index(chunks)