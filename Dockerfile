FROM python:3.12-slim
WORKDIR /srv
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Bake the embedding model into the image so first start needs no internet and no download wait.
RUN python -c "from chromadb.utils.embedding_functions import DefaultEmbeddingFunction as D; D()(['warm up'])"
COPY app ./app
COPY scripts ./scripts
COPY data ./data
COPY web ./web
COPY eval ./eval
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=40s CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
