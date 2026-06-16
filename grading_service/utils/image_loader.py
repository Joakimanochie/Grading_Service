import httpx
import base64
from typing import Tuple
from utils.logger import get_logger

logger = get_logger(__name__)


async def load_image_as_base64(image_url: str, response_id: str = "") -> Tuple[str, str]:
    """
    Fetches an image from responses.image_url and converts to base64.
    Used by handwritten_agent and diagram_agent.

    Returns: (base64_string, media_type)
    Raises: httpx.HTTPError if the URL is unreachable or returns non-200
    """
    logger.info(
        "Fetching image for grading",
        extra={"response_id": response_id, "url": image_url}
    )

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(image_url, timeout=15.0)
            response.raise_for_status()

        content_type = response.headers.get("content-type", "image/png")
        media_type = content_type.split(";")[0].strip()
        base64_data = base64.b64encode(response.content).decode("utf-8")

        logger.info(
            "Image fetched successfully",
            extra={"response_id": response_id, "media_type": media_type}
        )
        return base64_data, media_type

    except httpx.HTTPStatusError as e:
        logger.error(
            "Image fetch failed — HTTP error",
            extra={
                "response_id": response_id,
                "url": image_url,
                "status_code": e.response.status_code
            }
        )
        raise

    except httpx.RequestError as e:
        logger.error(
            "Image fetch failed — network error",
            extra={"response_id": response_id, "url": image_url, "error": str(e)}
        )
        raise
