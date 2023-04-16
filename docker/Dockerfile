FROM python:3.11-slim-bullseye as base

# Ensures that the container uses the UTF-8 character set. This is recommended for Python 3
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

LABEL name="transient"
LABEL maintainer="Jude Davis judedavis92@gmail.com"

RUN apt-get update && apt-get install -y --no-install-recommends \
  gcc=4:10.2.1-1 \
  libffi-dev=3.3-6 \
  linux-libc-dev=5.10.158-2 \
  build-essential=12.9 \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir pipenv

# Create and switch to a new user
RUN useradd --create-home appuser
WORKDIR /home/appuser
VOLUME /home/appuser
USER appuser

COPY ../Pipfile /home/appuser/Pipfile
COPY ../Pipfile.lock /home/appuser/Pipfile.lock

RUN pipenv install --dev

COPY ../ /home/appuser

# Disables the writing of .pyc files
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONPATH=/home/appuser/


