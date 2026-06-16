"""
Handwritten Agent — Groq vision reads a canvas handwriting image,
interprets the written content, and grades it against the marking guide.
"""

import os
import json
from models.request import ResponseItem
from models.response import GradeResult, FeedbackDetail
from utils.groq_client import get_groq_client
from utils.image_loader import load_image_as_base64
from utils.logger import get_logger

logger = get_logger(__name__)

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.65"))
MODEL = os.getenv("GROQ_MODEL_VISION", "meta-llama/llama-4-scout-17b-16e-instruct")


async def grade_handwritten(item: ResponseItem) -> GradeResult:
    client = get_groq_client()

    logger.info(
        "Handwritten grading started",
        extra={"response_id": item.response_id, "question_id": item.question_id}
    )

    try:
        image_data, media_type = await load_image_as_base64(
            item.student_image_url, item.response_id
        )
    except Exception as e:
        logger.error(
            "Handwritten agent — image load failed",
            extra={"response_id": item.response_id, "error": str(e)}
        )
        return _error_result(item, "Image could not be loaded — manual review required")

    prompt = f"""You are an expert academic examiner reviewing a student's handwritten answer.

QUESTION:
{item.question}

MARKING GUIDE (expected answer with mark allocations):
{item.expected_answer}

MAXIMUM MARKS: {item.max_marks}

The image provided is the student's handwritten answer on a digital canvas.
First read and interpret the handwriting carefully.
If completely illegible, set confidence to 0.1 and legibility_issue to true.
Then grade the interpreted answer strictly against the marking guide.

Respond ONLY with a valid JSON object — no preamble, no markdown backticks:
{{
  "score": <number between 0 and {item.max_marks}>,
  "confidence": <decimal 0.0 to 1.0>,
  "strengths": [<what the student answered correctly per the marking guide>],
  "weaknesses": [<what was missing or incorrect per the marking guide>],
  "legibility_issue": <true if handwriting was difficult to read, otherwise false>
}}"""

    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{image_data}"}
                    },
                    {"type": "text", "text": prompt}
                ]
            }],
            temperature=0.1,
        )
        raw = completion.choices[0].message.content.strip()
        result = json.loads(raw)

    except json.JSONDecodeError as e:
        logger.error(
            "Handwritten agent — invalid JSON from Groq",
            extra={"response_id": item.response_id, "error": str(e)}
        )
        return _error_result(item, "Grading failed — manual review required")

    except Exception as e:
        logger.error(
            "Handwritten agent — Groq call failed",
            extra={"response_id": item.response_id, "error": str(e)}
        )
        return _error_result(item, "Grading failed — manual review required")

    score = min(float(result.get("score", 0)), float(item.max_marks))
    confidence = float(result.get("confidence", 0.0))
    legibility_issue = result.get("legibility_issue", False)
    requires_review = confidence < CONFIDENCE_THRESHOLD or legibility_issue

    weaknesses = result.get("weaknesses", [])
    if legibility_issue:
        logger.warning(
            "Legibility issue detected",
            extra={"response_id": item.response_id}
        )
        weaknesses.append("Legibility issue detected — teacher review recommended")

    if requires_review:
        logger.warning(
            "Handwritten answer flagged for review",
            extra={
                "response_id": item.response_id,
                "confidence": confidence,
                "legibility_issue": legibility_issue,
            }
        )
    else:
        logger.info(
            "Handwritten graded",
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
            weaknesses=weaknesses,
        ),
        confidence=confidence,
        requires_review=requires_review,
        graded_by="ai",
    )


def _error_result(item: ResponseItem, message: str) -> GradeResult:
    return GradeResult(
        response_id=item.response_id,
        question_id=item.question_id,
        score=0.0,
        max_marks=item.max_marks,
        feedback=FeedbackDetail(strengths=[], weaknesses=[message]),
        confidence=0.0,
        requires_review=True,
        graded_by="ai",
    )
