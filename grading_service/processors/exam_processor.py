"""
Exam Processor — the batch queue orchestrator.

Algorithm:
  1. Receive all student sessions for one exam
  2. Split into chunks of CHUNK_SIZE (default 10)
  3. Process each chunk in parallel using asyncio.gather()
  4. Collect all graded sessions in order
  5. Return complete exam result

Example with 500 students, chunk size 10:
  Round 1:  students 1–10   processed in parallel
  Round 2:  students 11–20  processed in parallel
  ...
  Round 50: students 491–500 processed in parallel
  → Total: 50 rounds instead of 500 sequential operations
"""

import os
import asyncio
import time
from models.request import ExamGradeRequest, SessionRequest
from models.response import ExamGradeResponse, SessionResult, GradeResult, FeedbackDetail
from processors.session_grader import grade_session
from utils.logger import get_logger

logger = get_logger(__name__)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "10"))


async def process_exam(request: ExamGradeRequest) -> ExamGradeResponse:
    """
    Main entry point. Receives the full exam batch and returns all graded sessions.
    """
    total_sessions = len(request.sessions)
    start_time = time.time()

    logger.info(
        "Exam grading started",
        extra={
            "exam_id": request.exam_id,
            "total_sessions": total_sessions,
            "chunk_size": CHUNK_SIZE,
            "total_chunks": -(-total_sessions // CHUNK_SIZE),  # ceiling division
        }
    )

    chunks = _chunk_sessions(request.sessions, CHUNK_SIZE)
    all_results: list[SessionResult] = []

    for chunk_index, chunk in enumerate(chunks):
        logger.info(
            "Processing chunk",
            extra={
                "exam_id": request.exam_id,
                "chunk": chunk_index + 1,
                "total_chunks": len(chunks),
                "students_in_chunk": len(chunk),
            }
        )

        # Process all sessions in this chunk simultaneously
        chunk_results = await asyncio.gather(
            *[grade_session(session) for session in chunk],
            return_exceptions=True   # Don't let one failure kill the whole chunk
        )

        for i, result in enumerate(chunk_results):
            if isinstance(result, Exception):
                failed_session = chunk[i]
                logger.error(
                    "Session grading failed — returning empty result",
                    extra={
                        "exam_id": request.exam_id,
                        "session_id": failed_session.session_id,
                        "student_id": failed_session.student_id,
                        "error": str(result),
                    }
                )
                all_results.append(_failed_session_result(failed_session))
            else:
                all_results.append(result)

    duration = round(time.time() - start_time, 2)
    completed = len(all_results)

    logger.info(
        "Exam grading complete",
        extra={
            "exam_id": request.exam_id,
            "total_sessions": total_sessions,
            "completed_sessions": completed,
            "duration_seconds": duration,
            "flagged_sessions": sum(
                1 for s in all_results
                if any(g.requires_review for g in s.grades)
            ),
        }
    )

    return ExamGradeResponse(
        exam_id=request.exam_id,
        total_sessions=total_sessions,
        completed_sessions=completed,
        sessions=all_results,
    )


def _chunk_sessions(sessions: list[SessionRequest], size: int) -> list[list[SessionRequest]]:
    """
    Splits a flat list of sessions into sub-lists of `size`.

    Example: 25 sessions, size 10 → [[1..10], [11..20], [21..25]]
    """
    return [sessions[i:i + size] for i in range(0, len(sessions), size)]


def _failed_session_result(session: SessionRequest) -> SessionResult:
    """
    Fallback for a session that threw an unhandled exception.
    Returns all zeros, all flagged for review.
    """
    return SessionResult(
        session_id=session.session_id,
        student_id=session.student_id,
        total_score=0.0,
        max_score=sum(r.max_marks for r in session.responses),
        grades=[
            GradeResult(
                response_id=r.response_id,
                question_id=r.question_id,
                score=0.0,
                max_marks=r.max_marks,
                feedback=FeedbackDetail(
                    strengths=[],
                    weaknesses=["Session grading failed — manual review required"]
                ),
                confidence=0.0,
                requires_review=True,
                graded_by="ai",
            )
            for r in session.responses
        ],
    )
