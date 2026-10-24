# syntax=docker/dockerfile:1

# DeepFace (SFace + YuNet) runs on onnxruntime/OpenCV, so everything installs from prebuilt wheels:
# no compiler, no cmake, no multi-minute dlib build, no TensorFlow.
# (The old dlib engine, RECOGNIZER_BACKEND=dlib, needs a different image: see README.)
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        v4l-utils \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    DATA_DIR=/app/data \
    # model weights live in the image, so the running system never needs the internet
    DEEPFACE_HOME=/opt/deepface

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the model weights NOW (build time has internet). If you later change DEEPFACE_MODEL /
# DEEPFACE_DETECTOR in .env to something else, rebuild with the same values as build args:
#   docker compose build --build-arg DEEPFACE_MODEL=Facenet512
ARG DEEPFACE_MODEL=SFace
ARG DEEPFACE_DETECTOR=yunet
ENV DEEPFACE_MODEL=${DEEPFACE_MODEL} \
    DEEPFACE_DETECTOR=${DEEPFACE_DETECTOR}
RUN python -c "from deepface.modules import modeling; \
modeling.build_model(task='facial_recognition', model_name='${DEEPFACE_MODEL}'); \
modeling.build_model(task='face_detector', model_name='${DEEPFACE_DETECTOR}'); \
print('weights ready')"

COPY . .

# Created here as a fallback; docker-compose.yml normally mounts a volume
# over /app/data so this data survives rebuilds/restarts.
RUN mkdir -p /app/data

EXPOSE 8000

CMD ["python", "run.py"]
