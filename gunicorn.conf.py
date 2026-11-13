import os

bind = f"0.0.0.0:{os.environ.get('PORT', '8000')}"
workers = int(os.environ.get("WEB_CONCURRENCY", "2"))
timeout = 120
graceful_timeout = 30
accesslog = "-"
errorlog = "-"
capture_output = True
