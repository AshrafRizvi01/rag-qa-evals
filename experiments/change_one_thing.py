import re, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.rag import answer, load_docs, chunk, index, retrieve

GOLDEN = json.loads((ROOT / "data" / "golden.json").read_text())
CHUNK_SIZES = [300, 500, 800]

def key_tokens(expected):
    nums = re.findall(r"\d+(?:-|-)\d+|\d+%?|\d+", expected)
    return nums[:3] if nums else re.findall(r"[A-Za-z]{4,}", expected)[:2]

def score_chunk_size(size):
    chunks = []
    for doc in load_docs(str(ROOT / "docs")):
        chunks.extend(chunk(doc["text"], doc["source"], chunk_size=size))
    indexed = index(chunks)
    ret = facts = 0
    for item in GOLDEN:
        hits = retrieve(item["question"], indexed, k=3)
        if any(h["source"] == item["source"] for h in hits):
            ret += 1
        text, _ = answer(item["question"], indexed)
        toks = key_tokens(item["answer"])
        if toks and all(t.lower() in text.lower() for t in toks):
            facts += 1
    return size, ret, facts, len(GOLDEN)

if __name__ == "__main__":
    print("| chunk size | retrieval(top-3 source) | key-fact hit rate |")
    print("|-----------|------------------------|------------------|")
    for s in CHUNK_SIZES:
        size, r, f, n = score_chunk_size(s)
        print(f"| {size} | {r}/{n} | {f}/{n} |")
