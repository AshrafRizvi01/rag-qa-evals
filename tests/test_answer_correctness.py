from app.rag import answer

def test_refund_mentions_days(indexed):
    text, _ = answer("How many days do I have for a refund?", indexed)
    print(text)
    assert "7" in text or "10" in text

def test_deposit_value(indexed):
    text, _ = answer("What is the deposit needed for a booking?", indexed)
    print(text)
    assert "20" in text

def test_booking_full_payment_condition(indexed):
    text, _ = answer("What is the condition for full payment for a booking?", indexed)
    print(text)
    assert "15 days" in text

def test_out_of_scope_question(indexed):
    text, _ = answer("What is cancellation policy for sunset travels?", indexed)
    print(text)
    assert "I don't know" in text or "penalty" in text

