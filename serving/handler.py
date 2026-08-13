"""AWS Lambda entry point: the FastAPI app adapted by Mangum.

Local testing does not use this file; run the app with uvicorn instead
(`uvicorn serving.app:app`). Mangum is only the Lambda translation layer.
"""
from mangum import Mangum

from serving.app import app

handler = Mangum(app)
