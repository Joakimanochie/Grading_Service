import pytest
import json
from unittest.mock import patch
from agents.diagram_agent import grade_diagram
from tests.conftest import make_image_item, make_groq_response


@pytest.mark.asyncio
async def test_correct_diagram_returns_grade(mock_image_loader):
    mock_response = make_groq_response(json.dumps({
        "score": 4,
        "confidence": 0.80,
        "strengths": ["Nucleus present and labelled", "Cell wall drawn"],
        "weaknesses": ["Chloroplast missing"],
    }))

    with patch("agents.diagram_agent.get_groq_client") as mock_client:
        mock_client.return_value.chat.completions.create.return_value = mock_response
        result = await grade_diagram(make_image_item("diagram"))

    assert result.score == 4.0
    assert result.confidence == 0.80
    assert result.requires_review is False


@pytest.mark.asyncio
async def test_low_confidence_diagram_flagged(mock_image_loader):
    mock_response = make_groq_response(json.dumps({
        "score": 2,
        "confidence": 0.50,
        "strengths": [],
        "weaknesses": ["Most elements missing"],
    }))

    with patch("agents.diagram_agent.get_groq_client") as mock_client:
        mock_client.return_value.chat.completions.create.return_value = mock_response
        result = await grade_diagram(make_image_item("diagram"))

    assert result.requires_review is True
