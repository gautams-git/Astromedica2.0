FROM python:3.11.13-slim-bookworm

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser
WORKDIR /app

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

COPY . /app/
RUN mkdir -p /app/patients /app/charts_cache /app/rl_enhancer/data/uploads \
    && chown -R appuser:appuser /app

USER appuser
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    HOME=/tmp

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/', timeout=3)"

CMD ["gunicorn", "-c", "gunicorn.conf.py", "server:app"]