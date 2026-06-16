import pytest
import json
from unittest.mock import patch
from processors.session_grader import grade_session
from tests.conftest import make_session, make_mcq_item, make_theory_item, make_groq_response


@pytest.mark.asyncio
async def test_session_total_score_is_sum_of_grades():
    mcq = make_mcq_item(max_marks=2, selected_option_id="opt_A", correct_option_id="opt_A")
    theory = make_theory_item(max_marks=4)

    mock_response = make_groq_response(json.dumps({
        "score": 3, "confidence": 0.80, "strengths": [], "weaknesses": []
    }))

    with patch("agents.theory_agent.get_groq_client") as mock_client:
        mock_client.return_value.chat.completions.create.return_value = mock_response
        session = make_session(responses=[mcq, theory])
        result = await grade_session(session)

    assert result.total_score == 5.0   # 2 (mcq) + 3 (theory)
    assert result.max_score == 6       # 2 + 4
    assert len(result.grades) == 2


@pytest.mark.asyncio
async def test_empty_theory_answer_scores_zero():
    theory = make_theory_item(student_text=None)
    session = make_session(responses=[theory])
    result = await grade_session(session)

    assert result.total_score == 0.0
    assert result.grades[0].requires_review is False


@pytest.mark.asyncio
async def test_session_ids_preserved():
    session = make_session()
    with patch("agents.theory_agent.get_groq_client") as mock_client:
        mock_client.return_value.chat.completions.create.return_value = make_groq_response(
            json.dumps({"score": 2, "confidence": 0.7, "strengths": [], "weaknesses": []})
        )
        result = await grade_session(session)

    assert result.session_id == "sess_001"
    assert result.student_id == "stu_001"
