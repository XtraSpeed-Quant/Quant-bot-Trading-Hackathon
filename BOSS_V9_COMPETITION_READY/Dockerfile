FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN useradd --system --uid 10001 --create-home botuser \
    && mkdir -p /app/data /app/logs \
    && chown -R botuser:botuser /app
USER botuser

CMD ["python", "-u", "-m", "roostoo_bot"]
