from fastapi import FastAPI
from router import router
from utils.logger import init_sentry, get_logger

# Initialise Sentry before anything else
init_sentry()

logger = get_logger(__name__)

app = FastAPI(
    title="NitHub AI Grading Service",
    description="Batch AI grading microservice for the NitHub examination platform",
    version="2.0.0",
)

app.include_router(router)

logger.info("NitHub Grading Service started", extra={"version": "2.0.0"})
