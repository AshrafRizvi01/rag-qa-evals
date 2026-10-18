import re
import pytest

from app.rag import answer, ERROR_MESSAGE, REFUSAL_MESSAGE

def test_refund_mentions_days(indexed):
    text, _ = answer("How many days do I have for a refund?", indexed)
    print(text)
    assert re.search(r"\b7\b", text) and re.search(r"\b10\b", text)

def test_deposit_value(indexed):
    text, _ = answer("What is the deposit needed for a booking?", indexed)
    print(text)
    assert "20" in text

def test_booking_full_payment_condition(indexed):
    text, _ = answer("What is the condition for full payment for a booking?", indexed)
    print(text)
    assert "15 days" in text

def test_out_of_scope_question(indexed):
    text, _ = answer("What is cancellation policy for Hello travels?", indexed)
    print(text)
    assert "I don't know" in text

FALSE_REFUSAL_CASES = [
    pytest.param(
        "What are the rules for the refund?",
        [r"\b7\b", r"\b10\b"],
        id="refund-paraphrase",
    ),
    pytest.param(
        "How much must I pay upfront to confirm my booking as per the instructions?",
        [r"\b20\s*%"],
        id="deposit-paraphrase",
    ),
    pytest.param(
        "How early must I pay the remaining balance before travel?",
        [r"\b15\s+days\b"],
        id="balance-paraphrase",
    ),
]


@pytest.mark.parametrize("question, expected_patterns", FALSE_REFUSAL_CASES)
def test_answerable_question_not_refused(question, expected_patterns, indexed):
    text, hits = answer(question, indexed)
    details = (
        f"Question: {question!r}; answer: {text!r}; "
        f"sources: {[hit['source'] for hit in hits]}"
    )

    assert text.strip(), f"Empty answer. {details}"
    assert text.strip() != ERROR_MESSAGE, f"Model request failed. {details}"
    assert REFUSAL_MESSAGE.lower() not in text.lower(), (
        f"False refusal on an answerable question. {details}"
    )
    assert all(
        re.search(pattern, text, flags=re.IGNORECASE)
        for pattern in expected_patterns
    ), f"Answer is missing expected facts. {details}"

