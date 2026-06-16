import pytest
import json
from unittest.mock import patch
from processors.exam_processor import process_exam, _chunk_sessions
from tests.conftest import make_exam_request, make_session, make_mcq_item, make_groq_response


def test_chunk_sessions_splits_correctly():
    sessions = [make_session() for _ in range(25)]
    chunks = _chunk_sessions(sessions, 10)
    assert len(chunks) == 3
    assert len(chunks[0]) == 10
    assert len(chunks[1]) == 10
    assert len(chunks[2]) == 5


def test_chunk_sessions_single_student():
    sessions = [make_session()]
    chunks = _chunk_sessions(sessions, 10)
    assert len(chunks) == 1
    assert len(chunks[0]) == 1


@pytest.mark.asyncio
async def test_process_exam_returns_all_sessions():
    sessions = [
        make_session(responses=[make_mcq_item(
            response_id=f"resp_{i}",
            selected_option_id="opt_A",
            correct_option_id="opt_A"
        )])
        for i in range(3)
    ]
    request = make_exam_request(sessions=sessions)
    result = await process_exam(request)

    assert result.exam_id == "exam_001"
    assert result.total_sessions == 3
    assert result.completed_sessions == 3
    assert len(result.sessions) == 3


@pytest.mark.asyncio
async def test_process_exam_handles_one_failed_session():
    """If one session throws an error, others should still complete."""
    sessions = [make_session() for _ in range(3)]
    request = make_exam_request(sessions=sessions)

    call_count = 0

    async def mock_grade_session(session):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise Exception("Simulated session failure")
        from models.response import SessionResult
        return SessionResult(
            session_id=session.session_id,
            student_id=session.student_id,
            total_score=5.0,
            max_score=6,
            grades=[],
        )

    with patch("processors.exam_processor.grade_session", side_effect=mock_grade_session):
        result = await process_exam(request)

    assert result.completed_sessions == 3   # All 3 returned, failed one as fallback
