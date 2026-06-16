import pytest
import json
from unittest.mock import patch
from agents.handwritten_agent import grade_handwritten
from tests.conftest import make_image_item, make_groq_response


@pytest.mark.asyncio
async def test_normal_handwriting_returns_grade(mock_image_loader):
    mock_response = make_groq_response(json.dumps({
        "score": 4,
        "confidence": 0.75,
        "strengths": ["Answer clearly written"],
        "weaknesses": ["Missing one point"],
        "legibility_issue": False,
    }))

    with patch("agents.handwritten_agent.get_groq_client") as mock_client:
        mock_client.return_value.chat.completions.create.return_value = mock_response
        result = await grade_handwritten(make_image_item("handwritten"))

    assert result.score == 4.0
    assert result.requires_review is False


@pytest.mark.asyncio
async def test_legibility_issue_flags_review(mock_image_loader):
    mock_response = make_groq_response(json.dumps({
        "score": 1,
        "confidence": 0.3,
        "strengths": [],
        "weaknesses": [],
        "legibility_issue": True,
    }))

    with patch("agents.handwritten_agent.get_groq_client") as mock_client:
        mock_client.return_value.chat.completions.create.return_value = mock_response
        result = await grade_handwritten(make_image_item("handwritten"))

    assert result.requires_review is True
    assert any("legibility" in w.lower() for w in result.feedback.weaknesses)


@pytest.mark.asyncio
async def test_image_load_failure_returns_error_result(monkeypatch):
    async def bad_loader(url, response_id=""):
        raise Exception("404 Not Found")

    monkeypatch.setattr("agents.handwritten_agent.load_image_as_base64", bad_loader)
    result = await grade_handwritten(make_image_item("handwritten"))

    assert result.score == 0.0
    assert result.requires_review is True
