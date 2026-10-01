FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY roostoo_bot ./roostoo_bot
CMD ["python", "-m", "roostoo_bot"]
