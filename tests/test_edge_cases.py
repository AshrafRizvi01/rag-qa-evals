import pytest
from app.rag import answer

EDGE_CASES = [
    pytest.param("", id="empty-question"),
    pytest.param("What is the refund policy? " * 500, id="very-long-question"),
    pytest.param("返金について教えてください。何日かかりますか？", id="non-english-question"),
    pytest.param("What!@#$%^&*()_+ is the <script>refund</script> policy?", id="special-characters"),
]

@pytest.mark.parametrize("question", EDGE_CASES)
def test_answer_does_not_crash(question, indexed):
    text, hits = answer(question, indexed)
    assert isinstance(text, str)
    assert text.strip() != ""
