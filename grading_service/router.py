"""
API router — registers all endpoints.
POST /grade/exam  → main batch grading endpoint
GET  /health      → health check for Render and Node.js
"""

import time
from fastapi import APIRouter, HTTPException
from models.request import ExamGradeRequest
from models.response import ExamGradeResponse
from processors.exam_processor import process_exam
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post("/grade/exam", response_model=ExamGradeResponse)
async def grade_exam(request: ExamGradeRequest) -> ExamGradeResponse:
    """
    Main batch grading endpoint.
    Receives all student sessions for one exam.
    Returns all graded sessions in one response.

    Node.js makes ONE call here per exam submission.
    """
    logger.info(
        "Grade exam request received",
        extra={
            "exam_id": request.exam_id,
            "sessions": len(request.sessions),
        }
    )

    start = time.time()

    try:
        result = await process_exam(request)
        duration = round(time.time() - start, 2)

        logger.info(
            "Grade exam response sent",
            extra={
                "exam_id": request.exam_id,
                "completed_sessions": result.completed_sessions,
                "duration_seconds": duration,
            }
        )
        return result

    except Exception as e:
        logger.error(
            "Grade exam endpoint — unhandled error",
            extra={"exam_id": request.exam_id, "error": str(e)}
        )
        raise HTTPException(
            status_code=500,
            detail=f"Exam grading failed for exam {request.exam_id}: {str(e)}"
        )


@router.get("/health")
def health_check():
    """Health check endpoint. Render and Node.js ping this before sending requests."""
    return {"status": "ok", "service": "nithub-grading-service"}
