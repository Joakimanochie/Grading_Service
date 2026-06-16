from pydantic import BaseModel
from typing import List, Optional, Literal


class ResponseItem(BaseModel):
    """One question + student answer within a session."""
    response_id: str
    question_id: str
    question_type: Literal["mcq", "theory", "handwritten", "diagram"]
    response_type: str
    question: str
    expected_answer: str          # The marking guide set by the teacher per question
    max_marks: int

    # Student answer — only one of these will be populated per question type
    student_text: Optional[str] = None          # theory
    student_image_url: Optional[str] = None     # handwritten, diagram
    selected_option_id: Optional[str] = None    # mcq
    correct_option_id: Optional[str] = None     # mcq — Node.js resolves before calling


class SessionRequest(BaseModel):
    """One student's full exam session — all their questions and answers."""
    session_id: str
    student_id: str
    responses: List[ResponseItem]


class ExamGradeRequest(BaseModel):
    """
    The full exam batch request from Node.js.
    Contains ALL student sessions for one exam.
    Node.js makes ONE call with this payload.
    """
    exam_id: str
    sessions: List[SessionRequest]
