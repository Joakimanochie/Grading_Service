"""
API integration tests — tests the actual HTTP endpoints using FastAPI's test client.
No mocking of the endpoint layer — tests the full request/response cycle.
"""

import pytest
import json
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "nithub-grading-service"}


def test_grade_exam_valid_mcq_payload():
    payload = {
        "exam_id": "exam_test_001",
        "sessions": [
            {
                "session_id": "sess_001",
                "student_id": "stu_001",
                "responses": [
                    {
                        "response_id": "resp_001",
                        "question_id": "ques_001",
                        "question_type": "mcq",
                        "response_type": "text",
                        "question": "What is 2+2?",
                        "expected_answer": "4",
                        "max_marks": 2,
                        "selected_option_id": "opt_A",
                        "correct_option_id": "opt_A",
                    }
                ]
            }
        ]
    }

    response = client.post("/grade/exam", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["exam_id"] == "exam_test_001"
    assert data["total_sessions"] == 1
    assert data["completed_sessions"] == 1
    assert len(data["sessions"]) == 1
    assert data["sessions"][0]["total_score"] == 2.0


def test_grade_exam_missing_exam_id_returns_422():
    payload = {"sessions": []}
    response = client.post("/grade/exam", json=payload)
    assert response.status_code == 422


def test_grade_exam_invalid_question_type_returns_422():
    payload = {
        "exam_id": "exam_001",
        "sessions": [{
            "session_id": "sess_001",
            "student_id": "stu_001",
            "responses": [{
                "response_id": "resp_001",
                "question_id": "ques_001",
                "question_type": "invalid_type",   # not in Literal
                "response_type": "text",
                "question": "Q",
                "expected_answer": "A",
                "max_marks": 5,
            }]
        }]
    }
    response = client.post("/grade/exam", json=payload)
    assert response.status_code == 422


def test_grade_exam_response_shape():
    """Validates the full response shape Node.js expects."""
    payload = {
        "exam_id": "exam_shape_test",
        "sessions": [{
            "session_id": "sess_001",
            "student_id": "stu_001",
            "responses": [{
                "response_id": "resp_001",
                "question_id": "ques_001",
                "question_type": "mcq",
                "response_type": "text",
                "question": "Q",
                "expected_answer": "A",
                "max_marks": 1,
                "selected_option_id": "opt_A",
                "correct_option_id": "opt_B",
            }]
        }]
    }

    response = client.post("/grade/exam", json=payload)
    data = response.json()

    # Top level
    assert "exam_id" in data
    assert "total_sessions" in data
    assert "completed_sessions" in data
    assert "sessions" in data

    # Session level
    session = data["sessions"][0]
    assert "session_id" in session
    assert "student_id" in session
    assert "total_score" in session
    assert "max_score" in session
    assert "grades" in session

    # Grade level
    grade = session["grades"][0]
    assert "response_id" in grade
    assert "question_id" in grade
    assert "score" in grade
    assert "max_marks" in grade
    assert "feedback" in grade
    assert "strengths" in grade["feedback"]
    assert "weaknesses" in grade["feedback"]
    assert "confidence" in grade
    assert "requires_review" in grade
    assert "graded_by" in grade
