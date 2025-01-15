#!/bin/bash
redis-server --daemonize yes
celery -A app.processor.celery_app worker --loglevel=info &
uvicorn app.main:app --host 0.0.0.0 --port 8000