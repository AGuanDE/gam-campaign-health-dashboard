FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libpq-dev \
        postgresql-client && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /usr/app

COPY requirements.txt /usr/app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /usr/app/requirements.txt

# dbt will use /root/.dbt/profiles.yml by default, so no DBT_PROFILES_DIR needed
CMD ["bash"]
