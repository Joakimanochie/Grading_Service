"""
MCQ Agent — fully deterministic. No Groq call.
Node.js resolves correct_option_id from mcq_options.is_correct before calling.
"""

from models.request import ResponseItem
from models.response import GradeResult, FeedbackDetail
from utils.logger import get_logger

logger = get_logger(__name__)


def grade_mcq(item: ResponseItem) -> GradeResult:
    is_correct = (
        item.selected_option_id is not None
        and item.correct_option_id is not None
        and item.selected_option_id == item.correct_option_id
    )

    score = float(item.max_marks) if is_correct else 0.0

    logger.info(
        "MCQ graded",
        extra={
            "response_id": item.response_id,
            "question_id": item.question_id,
            "score": score,
            "max_marks": item.max_marks,
            "is_correct": is_correct,
        }
    )

    return GradeResult(
        response_id=item.response_id,
        question_id=item.question_id,
        score=score,
        max_marks=item.max_marks,
        feedback=FeedbackDetail(
            strengths=["Correct answer selected"] if is_correct else [],
            weaknesses=[] if is_correct else ["Incorrect answer selected"],
        ),
        confidence=1.0,
        requires_review=False,
        graded_by="ai",
    )
