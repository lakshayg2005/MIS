"""The groundedness metric must be able to FAIL: a metric that always says 100 % is worthless."""
from app.guardrails import sentence_groundedness

EVIDENCE = ["A security escort is mandatory for every woman employee on drops after 8:00 PM. The transport desk assigns the escort automatically when the drop is booked."]


def test_faithful_answers_score_fully_grounded():
    g = sentence_groundedness("An escort is mandatory for women on drops after 8 PM. The desk assigns it automatically.", EVIDENCE, "escort time?")
    assert g["grounded"] == g["sentences"] == 2 and g["ungrounded"] == []


def test_invented_number_is_caught():
    g = sentence_groundedness("An escort is mandatory for women on drops after 10 PM.", EVIDENCE)
    assert g["grounded"] == 0 and g["ungrounded"]


def test_invented_extra_claim_is_caught_even_without_numbers():
    g = sentence_groundedness("An escort is mandatory for women after 8 PM. Employees also receive complimentary breakfast vouchers and parking privileges.", EVIDENCE)
    assert g["sentences"] == 2 and g["grounded"] == 1 and "breakfast" in g["ungrounded"][0]


def test_off_topic_answer_is_caught():
    g = sentence_groundedness("Bananas are rich in potassium and grow in tropical regions around the world.", EVIDENCE)
    assert g["grounded"] == 0


def test_numbers_from_the_question_are_allowed():
    g = sentence_groundedness("Your 9:30 PM drop is after 8 PM, so an escort is mandatory.", EVIDENCE, "I leave at 9:30 PM, is an escort needed?")
    assert g["grounded"] == g["sentences"]


def test_short_confirmation_sentence_is_not_flagged():
    assert sentence_groundedness("Yes.", EVIDENCE)["ungrounded"] == []
