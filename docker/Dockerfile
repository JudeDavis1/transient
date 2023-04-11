FROM python:3.11-slim-bullseye as base

# Ensures that the container uses the UTF-8 character set. This is recommended for Python 3
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8


# Disables the writing of .pyc files
ENV PYTHONDONTWRITEBYTECODE 1

FROM base AS builder

LABEL name="transient"
LABEL maintainer="Jude Davis judedavis92@gmail.com"

WORKDIR /

COPY ./ ./

# Install pipenv and packages
RUN apt-get update && apt-get upgrade -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
FROM builder AS python-deps

RUN pip install --upgrade pip
RUN pip install pipenv && pipenv install --dev

FROM base AS runtime

# Create and switch to a new user
RUN useradd --create-home appuser
WORKDIR /home/appuser
VOLUME /home/appuser
USER appuser


# Run the application
# You can for example test with `docker run (container name) pipenv run test`

