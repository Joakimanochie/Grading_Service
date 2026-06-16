import pytest
from agents.mcq_agent import grade_mcq
from tests.conftest import make_mcq_item


def test_correct_answer_full_marks():
    item = make_mcq_item(selected_option_id="opt_A", correct_option_id="opt_A")
    result = grade_mcq(item)
    assert result.score == 2.0
    assert result.confidence == 1.0
    assert result.requires_review is False
    assert "Correct" in result.feedback.strengths[0]


def test_wrong_answer_zero_marks():
    item = make_mcq_item(selected_option_id="opt_B", correct_option_id="opt_A")
    result = grade_mcq(item)
    assert result.score == 0.0
    assert result.confidence == 1.0
    assert result.requires_review is False
    assert len(result.feedback.weaknesses) > 0


def test_missing_selected_option_zero_marks():
    item = make_mcq_item(selected_option_id=None, correct_option_id="opt_A")
    result = grade_mcq(item)
    assert result.score == 0.0


def test_missing_correct_option_zero_marks():
    item = make_mcq_item(selected_option_id="opt_A", correct_option_id=None)
    result = grade_mcq(item)
    assert result.score == 0.0


def test_graded_by_is_ai():
    item = make_mcq_item()
    result = grade_mcq(item)
    assert result.graded_by == "ai"
