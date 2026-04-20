FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
EXPOSE 8080

# Hosted browser mode only. The local operator app (app.webapp) is
# not exposed by this image — it needs the ESP32 on a USB port,
# which doesn't translate to a container.
CMD ["python", "-m", "app.web_app", "--host", "0.0.0.0", "--port", "8080"]
