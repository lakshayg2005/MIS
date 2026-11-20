"""Cross-encoder reranking: score (question, passage) pairs TOGETHER, which is more accurate than comparing two
independent vectors, but slower, so we only apply it to the few candidates the cheap search already found."""


class CrossEncoderReranker:
    def __init__(self, model: str = "Xenova/ms-marco-MiniLM-L-6-v2"):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._ce = TextCrossEncoder(model)

    def rerank(self, question: str, hits: list) -> list:
        if len(hits) < 2:
            return hits
        scores = list(self._ce.rerank(question, [f"{h['title']}. {h['text']}" for h in hits]))
        return [h for _, h in sorted(zip(scores, hits), key=lambda x: -x[0])]
