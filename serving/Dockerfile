# AWS Lambda container image for the credit-risk scoring API.
# A container (not a zip) is required because lightgbm + pandas exceed the Lambda
# zip size limit. The model is NOT baked in: the Lambda streams it from S3 at cold
# start (MODEL_S3_URI), so this image is model-free and CI can build it.
# Python 3.13 so the latest numpy/pandas ship prebuilt wheels (the 3.11 image does
# not, and the base image has no C compiler to build them from source).
FROM public.ecr.aws/lambda/python:3.13

WORKDIR ${LAMBDA_TASK_ROOT}

# Install the package with its serving extras (FastAPI, Pydantic, Mangum, boto3).
# Upgrade pip first so it recognizes current wheel tags.
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir ".[serving]"

# Application code (no model artifact).
COPY serving ./serving

# Mangum handler: module serving.handler, object handler.
CMD ["serving.handler.handler"]
