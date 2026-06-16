import pytest
import json
from unittest.mock import patch
from agents.theory_agent import grade_theory
from tests.conftest import make_theory_item, make_groq_response


def test_normal_answer_returns_grade():
    mock_response = make_groq_response(json.dumps({
        "score": 3,
        "confidence": 0.85,
        "strengths": ["Mentioned sunlight"],
        "weaknesses": ["Did not mention glucose"],
    }))

    with patch("agents.theory_agent.get_groq_client") as mock_client:
        mock_client.return_value.chat.completions.create.return_value = mock_response
        result = grade_theory(make_theory_item())

    assert result.score == 3.0
    assert result.confidence == 0.85
    assert result.requires_review is False
    assert result.graded_by == "ai"


def test_score_never_exceeds_max_marks():
    mock_response = make_groq_response(json.dumps({
        "score": 999,
        "confidence": 0.9,
        "strengths": [],
        "weaknesses": [],
    }))

    with patch("agents.theory_agent.get_groq_client") as mock_client:
        mock_client.return_value.chat.completions.create.return_value = mock_response
        result = grade_theory(make_theory_item(max_marks=4))

    assert result.score == 4.0


def test_low_confidence_flags_review():
    mock_response = make_groq_response(json.dumps({
        "score": 2,
        "confidence": 0.4,
        "strengths": [],
        "weaknesses": [],
    }))

    with patch("agents.theory_agent.get_groq_client") as mock_client:
        mock_client.return_value.chat.completions.create.return_value = mock_response
        result = grade_theory(make_theory_item())

    assert result.requires_review is True


def test_invalid_json_from_groq_returns_zero():
    mock_response = make_groq_response("This is not JSON at all")

    with patch("agents.theory_agent.get_groq_client") as mock_client:
        mock_client.return_value.chat.completions.create.return_value = mock_response
        result = grade_theory(make_theory_item())

    assert result.score == 0.0
    assert result.requires_review is True
    assert result.confidence == 0.0


def test_groq_exception_returns_zero():
    with patch("agents.theory_agent.get_groq_client") as mock_client:
        mock_client.return_value.chat.completions.create.side_effect = Exception("Groq timeout")
        result = grade_theory(make_theory_item())

    assert result.score == 0.0
    assert result.requires_review is True
