FROM python:3.11-slim-bullseye as base

# Ensures that the container uses the UTF-8 character set. This is recommended for Python 3
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8


# Disables the writing of .pyc files
ENV PYTHONDONTWRITEBYTECODE 1

FROM base AS builder

LABEL name="transient"
LABEL maintainer="Jude Davis judedavis92@gmail.com"

WORKDIR /transient

COPY ./ ./

# Install pipenv and packages
RUN apt update && apt upgrade -y
FROM builder AS python-deps

RUN pip install pipenv && pipenv install --dev

FROM base AS runtime

# Run the application
# You can for example test with `docker run (container name) pipenv run test`

