"""
Shared fixtures and mock factories for all tests.
Mocks Groq and httpx so tests run without real API calls.
"""

import pytest
from unittest.mock import MagicMock
from models.request import ResponseItem, SessionRequest, ExamGradeRequest


# ── Model Factories ───────────────────────────────────────────────────────────

def make_mcq_item(**kwargs) -> ResponseItem:
    defaults = dict(
        response_id="resp_mcq_001",
        question_id="ques_001",
        question_type="mcq",
        response_type="text",
        question="What is 2 + 2?",
        expected_answer="4",
        max_marks=2,
        selected_option_id="opt_A",
        correct_option_id="opt_A",
    )
    return ResponseItem(**{**defaults, **kwargs})


def make_theory_item(**kwargs) -> ResponseItem:
    defaults = dict(
        response_id="resp_theory_001",
        question_id="ques_002",
        question_type="theory",
        response_type="text",
        question="Explain photosynthesis.",
        expected_answer="Award 2 marks for light energy, 2 for glucose production.",
        max_marks=4,
        student_text="Plants use sunlight to make food from CO2 and water.",
    )
    return ResponseItem(**{**defaults, **kwargs})


def make_image_item(question_type="diagram", **kwargs) -> ResponseItem:
    defaults = dict(
        response_id="resp_img_001",
        question_id="ques_003",
        question_type=question_type,
        response_type=question_type,
        question="Draw and label a plant cell.",
        expected_answer="2 marks nucleus, 2 marks cell wall, 1 mark chloroplast.",
        max_marks=5,
        student_image_url="https://storage.example.com/canvas/img.png",
    )
    return ResponseItem(**{**defaults, **kwargs})


def make_session(responses=None) -> SessionRequest:
    if responses is None:
        responses = [make_mcq_item(), make_theory_item()]
    return SessionRequest(
        session_id="sess_001",
        student_id="stu_001",
        responses=responses,
    )


def make_exam_request(sessions=None) -> ExamGradeRequest:
    if sessions is None:
        sessions = [make_session()]
    return ExamGradeRequest(exam_id="exam_001", sessions=sessions)


# ── Groq Mock Factory ─────────────────────────────────────────────────────────

def make_groq_response(json_string: str):
    """Creates a mock Groq completion response containing the given JSON string."""
    mock_choice = MagicMock()
    mock_choice.message.content = json_string
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    return mock_completion


# ── Image Loader Mock ─────────────────────────────────────────────────────────

@pytest.fixture
def mock_image_loader(monkeypatch):
    """Patches load_image_as_base64 at each agent's import site so no real HTTP call is made."""
    async def fake_loader(url, response_id=""):
        return "base64encodedimagedata", "image/png"

    monkeypatch.setattr("agents.handwritten_agent.load_image_as_base64", fake_loader)
    monkeypatch.setattr("agents.diagram_agent.load_image_as_base64", fake_loader)
