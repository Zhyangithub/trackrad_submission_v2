FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /opt/app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH=/opt/app
