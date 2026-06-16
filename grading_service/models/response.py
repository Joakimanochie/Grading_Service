from pydantic import BaseModel
from typing import List


class FeedbackDetail(BaseModel):
    strengths: List[str]
    weaknesses: List[str]


class GradeResult(BaseModel):
    """Grading result for one question response."""
    response_id: str
    question_id: str
    score: float
    max_marks: int
    feedback: FeedbackDetail
    confidence: float           # 0.00 – 1.00
    requires_review: bool       # True if confidence < CONFIDENCE_THRESHOLD
    graded_by: str              # Always "ai"


class SessionResult(BaseModel):
    """Grading results for one student's full session."""
    session_id: str
    student_id: str
    total_score: float          # SUM of all GradeResult.score
    max_score: int              # SUM of all GradeResult.max_marks
    grades: List[GradeResult]


class ExamGradeResponse(BaseModel):
    """
    The full exam batch response back to Node.js.
    One response containing ALL graded student sessions.
    """
    exam_id: str
    total_sessions: int         # How many sessions were received
    completed_sessions: int     # How many were successfully graded
    sessions: List[SessionResult]
