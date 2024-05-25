FROM python:3.11-slim

WORKDIR /workspaces
COPY web-service/requirements.txt /workspaces

RUN pip install -r requirements.txt

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY web-service/app.py /workspaces
COPY exercise/mlruns /workspaces/project-bazaaer/exercise/mlruns

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "app:app"]