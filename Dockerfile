FROM harbor.4pd.io/sagegpt-aio/pk_platform/ubuntu_python:24.04_3.12

WORKDIR /app
COPY pyproject.toml .
COPY src ./src

RUN pip install --no-cache-dir .

EXPOSE 8080
USER 65534:65534

ENTRYPOINT ["python3", "-m", "decision_gen"]
