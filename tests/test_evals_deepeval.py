import json
import os

import pytest
from deepeval.test_case import LLMTestCase
from deepeval.models import OpenAIModel
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from app.rag import answer

pytestmark = pytest.mark.deepeval

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "golden.json")
with open(GOLDEN_PATH, "r") as f:
    goldens = json.load(f)

results = []

@pytest.fixture(scope="module", autouse=True)
def print_results_summary():
    yield
    print("\n=== Evaluation Summary ===")
    for r in results:
        print(f"Q: {r['question']}")
        print(f"A: {r['answer']}")
        print(f"AnswerRelevancy: {r['answer_relevancy']}, Faithfulness: {r['faithfulness']}\n")

def measure_with_retry(metric_cls, threshold, case, attempts=2):
    """Retry with a nonzero judge temperature: at temperature=0 a same-input
    retry just reproduces the same degenerate/looping output deterministically."""
    last_error = None
    metric = None
    for attempt in range(attempts):
        judge = OpenAIModel(
            model="gpt-4o-mini",
            generation_kwargs={"max_completion_tokens": 8192},
            temperature=0 if attempt == 0 else 0.5,
        )
        metric = metric_cls(threshold=threshold, model=judge)
        try:
            metric.measure(case)
            return metric, None
        except Exception as e:
            last_error = e
    return metric, f"{metric_cls.__name__} crashed after {attempts} attempts: {last_error}"

@pytest.mark.parametrize("row", goldens, ids=lambda row: row["question"])
def test_refund_relevancy_and_faithfulness(indexed, row):
    question = row["question"]
    text, hits = answer(question, indexed)
    case = LLMTestCase(
        input=question,
        actual_output=text,
        expected_output=row["answer"],
        retrieval_context=[hit["text"] for hit in hits],
    )
    relevancy, relevancy_error = measure_with_retry(AnswerRelevancyMetric, 0.5, case)
    faithfulness, faithfulness_error = measure_with_retry(FaithfulnessMetric, 0.5, case)

    errors = [error for error in (relevancy_error, faithfulness_error) if error]

    results.append({
        "question": question,
        "answer": text,
        "answer_relevancy": relevancy.score,
        "faithfulness": faithfulness.score,
    })

    assert not errors, "; ".join(errors)
    assert relevancy.is_successful(), relevancy.reason
    assert faithfulness.is_successful(), faithfulness.reason