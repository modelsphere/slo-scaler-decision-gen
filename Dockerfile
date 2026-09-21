# The base image is a build arg so that a build behind a registry mirror, or on
# a runner that cannot reach Docker Hub, can point it at its own registry.
# See .gitlab-ci.yml for how the internal CI overrides it.
ARG BASE_IMAGE=python:3.12-slim

FROM ${BASE_IMAGE}

# Optional PyPI mirror, for the same reason. Empty (the default) uses PyPI.
#   docker build --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple .
ARG PIP_INDEX_URL=""

WORKDIR /app
COPY pyproject.toml .
COPY src ./src

# Turning the progress bar off is not about log noise: a CI runner container can
# have a low thread limit, and pip's default rich progress bar starts a refresh
# thread, which fails the whole install with
# `RuntimeError: can't start new thread`. It does not reproduce in a local
# docker build -- only on the runner.
#
# It has to be the environment variable, not the flag. Installing this project
# runs a PEP 517 build, and pip spawns a NESTED pip to install the build
# dependencies; that child does not inherit command-line flags, only the
# environment. The flag alone still fails, in the child.
ENV PIP_PROGRESS_BAR=off

RUN if [ -n "$PIP_INDEX_URL" ]; then \
        pip install --no-cache-dir -i "$PIP_INDEX_URL" . ; \
    else \
        pip install --no-cache-dir . ; \
    fi

EXPOSE 8080
USER 65534:65534

ENTRYPOINT ["python3", "-m", "decision_gen"]
