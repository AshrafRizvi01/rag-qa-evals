import pytest
from app.rag import answer, PROMPT_EMPTY, PROMPT_TOO_LONG, MAX_QUESTION_LENGTH

EDGE_CASES = [
    pytest.param("返金について教えてください。何日かかりますか？", id="non-english-question"),
    pytest.param("What!@#$%^&*()_+ is the <script>refund</script> policy?", id="special-characters"),
]

@pytest.mark.parametrize("question", EDGE_CASES)
def test_answer_does_not_crash(question, indexed):
    text, hits = answer(question, indexed)
    assert isinstance(text, str)
    assert text.strip() != ""

def test_empty_question_is_rejected(indexed):
    text, hits = answer("", indexed)
    assert text == PROMPT_EMPTY
    assert hits == []

def test_very_long_question_is_rejected(indexed):
    question = "What is the refund policy? " * 500
    assert len(question) > MAX_QUESTION_LENGTH
    text, hits = answer(question, indexed)
    assert text == PROMPT_TOO_LONG
    assert hits == []

