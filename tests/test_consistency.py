import re
from app.rag import answer
import pytest

def _refund_window(text):
    if re.search(r"\b7\b", text) and re.search(r"\b10\b", text):
        return "7-10"
    return None

def _passport_validity(text):
    if re.search(r"\b6\b", text) and re.search(r"month", text, re.I):
        return "6-months"
    return None

def _deposit_pct(text):
    if re.search(r"\b20\s*%", text) or re.search(r"\b20\b.{0,20}\bpercent", text, re.I):
        return "20%"
    return None

CASES = [
    ("refund window", ["How many days I have for refund?", "How long does an approve refund take?", "When will I get my money back after refund?"], _refund_window, "7-10"),
    ("passport validity", ["How long should my passport be valid?", "What is the minimum passport validity required?"], _passport_validity, "6-months"),
    ("deposit percentage", ["What is the deposit needed for a booking?", "How much deposit is required?"], _deposit_pct, "20%"),
]

REPEAT_N = 3

@pytest.mark.parametrize("case_id, questions, extractor, expected", CASES, ids=[c[0] for c in CASES])
def test_paraphrases_same_fact(case_id, questions, extractor, expected, indexed):
    facts = []
    for q in questions:
        text, _ = answer(q, indexed)
        f = extractor(text)
        assert f is not None, f"missing key fact for {q!r}: {text!r}"
        facts.append(f)
        assert all(f == expected for f in facts), f"{case_id}: got {facts}"

@pytest.mark.parametrize("case_id, questions, extractor, expected", CASES, ids=[c[0] for c in CASES])
def test_repeat_same_fact(case_id, questions, extractor, expected, indexed):
    facts = []
    for i in range(REPEAT_N):
        text, _ = answer(questions[0], indexed)
        f = extractor(text)
        assert f is not None, f"missing key fact for {questions[0]!r}: {text!r}"
        facts.append(f)
        assert all(f == expected for f in facts), f"{case_id}: repeats {facts}"
