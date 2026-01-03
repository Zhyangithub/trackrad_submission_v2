FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# ===== 1. 创建非 root 用户 =====
RUN groupadd -r appuser && useradd -m -r -g appuser appuser

WORKDIR /opt/app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH=/opt/app

# ===== 2. 切换用户（关键！）=====
USER appuser
