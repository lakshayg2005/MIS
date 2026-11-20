"""Metric definitions (pure functions, unit-tested). Everything is deterministic except the optional judge.

RETRIEVAL (per answerable case; relevant = the expected sections, retrieved = sections of the top-k chunks)
  recall@k    |relevant ∩ top-k| / |relevant|      did we find what we needed?
  precision@k |relevant ∩ top-k| / k               how much of what we show the LLM is useful? (bounded by
                                                    |relevant|/k, so it is a relative number: compare modes, not to 1.0)
  hit@k       1 if any relevant section is in the top-k
  MRR         1 / rank of the first relevant section
  nDCG@k      rank-aware gain with binary relevance (1.0 = all relevant sections ranked first)

ANSWER QUALITY / HALLUCINATION (per answered case)
  fact_recall          share of required facts present in the answer
  citation_precision   share of cited sections that are expected      (are citations on point?)
  citation_recall      share of expected sections that were cited
  numeric_hallucination 1 if the answer has a number found in neither the cited text nor the question
  support_ratio        share of the answer's content words found in the cited text (low = unsupported wording)
  judge_faithful/judge_relevant (optional, LLM judge): every claim supported by the excerpts? answers the question?
SYSTEM LEVEL
  abstention recall    share of not-covered questions where we correctly said "not covered"  (1 - false-answer rate)
  abstention precision share of "not covered" replies that were right
  over-abstention      share of answerable questions we refused
  leak rate            cases with a citation from another company or forbidden text
"""
import math


def recall_at_k(relevant: list, got: list, k: int) -> float:
    return len(set(relevant) & set(got[:k])) / len(set(relevant)) if relevant else 0.0


def precision_at_k(relevant: list, got: list, k: int) -> float:
    return len(set(relevant) & set(got[:k])) / k


def hit_at_k(relevant: list, got: list, k: int) -> float:
    return float(bool(set(relevant) & set(got[:k])))


def reciprocal_rank(relevant: list, got: list) -> float:
    return next((1.0 / (i + 1) for i, g in enumerate(got) if g in relevant), 0.0)


def ndcg_at_k(relevant: list, got: list, k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, g in enumerate(got[:k]) if g in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(set(relevant)), k)))
    return dcg / ideal if ideal else 0.0


def mean(xs: list):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None
