FROM python:3.11-slim

# Install Redis
RUN apt-get update && apt-get install -y redis-server

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY start.sh .
RUN chmod +x start.sh

EXPOSE 8000
CMD ["./start.sh"]