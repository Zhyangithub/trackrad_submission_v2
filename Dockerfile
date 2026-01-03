FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /opt/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY algorithm ./algorithm

# ---- create non-root user (REQUIRED) ----
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /opt/app
USER appuser

ENV PYTHONPATH=/opt/app
