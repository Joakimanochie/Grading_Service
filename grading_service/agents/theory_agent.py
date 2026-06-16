"""
Theory Agent — Groq LLM evaluates student text against the marking guide.
"""

import os
import json
from models.request import ResponseItem
from models.response import GradeResult, FeedbackDetail
from utils.groq_client import get_groq_client
from utils.logger import get_logger

logger = get_logger(__name__)

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.65"))
MODEL = os.getenv("GROQ_MODEL_TEXT", "meta-llama/llama-4-scout-17b-16e-instruct")


def grade_theory(item: ResponseItem) -> GradeResult:
    client = get_groq_client()

    logger.info(
        "Theory grading started",
        extra={"response_id": item.response_id, "question_id": item.question_id}
    )

    prompt = f"""You are an expert academic examiner. Grade the student's answer strictly according to the marking guide provided by the teacher.

QUESTION:
{item.question}

MARKING GUIDE (expected answer with mark allocations):
{item.expected_answer}

MAXIMUM MARKS: {item.max_marks}

STUDENT ANSWER:
{item.student_text or "[No answer provided]"}

Grade ONLY based on the marking guide. Do not award marks for correct information not in the marking guide.

Respond ONLY with a valid JSON object — no preamble, no markdown backticks:
{{
  "score": <number between 0 and {item.max_marks}>,
  "confidence": <decimal 0.0 to 1.0 — your certainty in this grade>,
  "strengths": [<strings — what the student answered correctly per the marking guide>],
  "weaknesses": [<strings — what was missing or incorrect per the marking guide>]
}}"""

    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        raw = completion.choices[0].message.content.strip()
        result = json.loads(raw)

    except json.JSONDecodeError as e:
        logger.error(
            "Theory agent — Groq returned invalid JSON",
            extra={"response_id": item.response_id, "error": str(e)}
        )
        result = {
            "score": 0,
            "confidence": 0.0,
            "strengths": [],
            "weaknesses": ["Grading failed — manual review required"],
        }

    except Exception as e:
        logger.error(
            "Theory agent — Groq call failed",
            extra={"response_id": item.response_id, "error": str(e)}
        )
        result = {
            "score": 0,
            "confidence": 0.0,
            "strengths": [],
            "weaknesses": ["Grading failed — manual review required"],
        }

    score = min(float(result.get("score", 0)), float(item.max_marks))
    confidence = float(result.get("confidence", 0.0))
    requires_review = confidence < CONFIDENCE_THRESHOLD

    if requires_review:
        logger.warning(
            "Low confidence — flagged for review",
            extra={
                "response_id": item.response_id,
                "confidence": confidence,
                "threshold": CONFIDENCE_THRESHOLD,
            }
        )
    else:
        logger.info(
            "Theory graded",
            extra={
                "response_id": item.response_id,
                "score": score,
                "max_marks": item.max_marks,
                "confidence": confidence,
            }
        )

    return GradeResult(
        response_id=item.response_id,
        question_id=item.question_id,
        score=score,
        max_marks=item.max_marks,
        feedback=FeedbackDetail(
            strengths=result.get("strengths", []),
            weaknesses=result.get("weaknesses", []),
        ),
        confidence=confidence,
        requires_review=requires_review,
        graded_by="ai",
    )
