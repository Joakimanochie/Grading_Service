"""
Session Grader — grades one student's full session.
Called by exam_processor for each student in the batch.
Loops through all responses in order and calls the correct agent per question type.
"""

from models.request import SessionRequest, ResponseItem
from models.response import SessionResult, GradeResult, FeedbackDetail
from agents.mcq_agent import grade_mcq
from agents.theory_agent import grade_theory
from agents.handwritten_agent import grade_handwritten
from agents.diagram_agent import grade_diagram
from utils.logger import get_logger

logger = get_logger(__name__)


async def grade_session(session: SessionRequest) -> SessionResult:
    """
    Grades all responses in one student session sequentially.
    Returns a SessionResult with all individual grades and the total score.
    """
    logger.info(
        "Session grading started",
        extra={
            "session_id": session.session_id,
            "student_id": session.student_id,
            "questions": len(session.responses),
        }
    )

    grades = []

    for item in session.responses:
        grade = await _grade_single_response(item)
        grades.append(grade)

    total_score = sum(g.score for g in grades)
    max_score = sum(g.max_marks for g in grades)

    logger.info(
        "Session grading complete",
        extra={
            "session_id": session.session_id,
            "student_id": session.student_id,
            "total_score": total_score,
            "max_score": max_score,
            "flagged": sum(1 for g in grades if g.requires_review),
        }
    )

    return SessionResult(
        session_id=session.session_id,
        student_id=session.student_id,
        total_score=total_score,
        max_score=max_score,
        grades=grades,
    )


async def _grade_single_response(item: ResponseItem) -> GradeResult:
    """
    Routes one response to the correct agent.
    Returns an error result (score 0, requires_review True) on any failure.
    """
    question_type = item.question_type

    try:
        # Empty answer check — short-circuit before any agent or Groq call
        if question_type == "theory" and not item.student_text:
            return _empty_result(item)

        if question_type in ("handwritten", "diagram") and not item.student_image_url:
            return _empty_result(item)

        if question_type == "mcq":
            return grade_mcq(item)                   # sync — no await
        elif question_type == "theory":
            return grade_theory(item)                # sync — no await
        elif question_type == "handwritten":
            return await grade_handwritten(item)     # async — await
        elif question_type == "diagram":
            return await grade_diagram(item)         # async — await
        else:
            logger.error(
                "Unknown question type",
                extra={"response_id": item.response_id, "question_type": question_type}
            )
            return _empty_result(item, f"Unknown question type: {question_type}")

    except Exception as e:
        logger.error(
            "Unhandled error grading response",
            extra={"response_id": item.response_id, "error": str(e)}
        )
        return GradeResult(
            response_id=item.response_id,
            question_id=item.question_id,
            score=0.0,
            max_marks=item.max_marks,
            feedback=FeedbackDetail(
                strengths=[],
                weaknesses=["Grading error — manual review required"]
            ),
            confidence=0.0,
            requires_review=True,
            graded_by="ai",
        )


def _empty_result(item: ResponseItem, reason: str = "No answer provided") -> GradeResult:
    """Zero score for unanswered questions. No agent or Groq call made."""
    logger.info(
        "Empty answer — skipping agent",
        extra={"response_id": item.response_id, "reason": reason}
    )
    return GradeResult(
        response_id=item.response_id,
        question_id=item.question_id,
        score=0.0,
        max_marks=item.max_marks,
        feedback=FeedbackDetail(strengths=[], weaknesses=[reason]),
        confidence=1.0,
        requires_review=False,
        graded_by="ai",
    )
