"""Report-generation metrics, matching EchoSonar-R's set (arXiv 2606.28164).

Quote from that paper's metric section:
  "We use natural language generation (NLG) metrics: BLEU-1/2/3/4, which measure
   n-gram precision between generated and reference reports; METEOR, which extends
   n-gram matching with synonym recognition and stemming; and ROUGE-L, which
   captures the longest common subsequence between candidate and reference texts."
plus BERTScore, plus GREEN. CIDEr is NOT used.

Implemented here: BLEU-1..4 (with the standard brevity penalty) and ROUGE-L, in
stdlib, because they are deterministic and cheap. BERTScore and GREEN need models
and live elsewhere:
  - BERTScore: a contextual embedding model.
  - GREEN [Ostmeier et al., arXiv 2405.03595]: LLM-judged clinical faithfulness
    from counts of significant errors, omissions and matched findings. EchoSonar-R
    uses an echocardiography-ADAPTED GREEN prompt judged by Mistral-7B, and defers
    the formula to the original paper -- so our GREEN is NOT implemented here and
    must not be approximated with a lexical stand-in. Reporting a home-made
    "GREEN" next to their 0.800 would be a false comparison.

METEOR is also omitted rather than faked: it needs synonym/stemming resources
(WordNet), and a home-rolled version would not be their METEOR.

These are corpus-level by default, matching how BLEU is normally reported.
"""
import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list:
    return _TOKEN.findall((text or "").lower())


def _ngrams(tokens: list, n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def bleu(predictions: list, references: list, *, max_n: int = 4) -> float:
    """Corpus BLEU-`max_n` with uniform weights and the standard brevity penalty.

    Clipped n-gram precision: a repeated n-gram can only be credited as often as
    it appears in the reference, so padding the output with a common phrase does
    not inflate the score.
    """
    if not predictions:
        return 0.0
    num = [0] * max_n
    den = [0] * max_n
    pred_len = ref_len = 0
    for pred, ref in zip(predictions, references):
        p_tok, r_tok = tokenize(pred), tokenize(ref)
        pred_len += len(p_tok)
        ref_len += len(r_tok)
        for n in range(1, max_n + 1):
            p_ng, r_ng = _ngrams(p_tok, n), _ngrams(r_tok, n)
            den[n - 1] += max(0, len(p_tok) - n + 1)
            num[n - 1] += sum(min(c, r_ng[g]) for g, c in p_ng.items())
    if any(d == 0 for d in den) or any(x == 0 for x in num):
        return 0.0
    log_p = sum(math.log(num[i] / den[i]) for i in range(max_n)) / max_n
    bp = 1.0 if pred_len > ref_len else math.exp(1 - ref_len / max(pred_len, 1))
    return bp * math.exp(log_p)


def bleu_n(predictions: list, references: list) -> dict:
    """BLEU-1 through BLEU-4, as EchoSonar-R's Table 3 reports them."""
    return {f"BLEU-{n}": bleu(predictions, references, max_n=n) for n in (1, 2, 3, 4)}


def _lcs_length(a: list, b: list) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(cur[j], prev[j + 1]))
        prev = cur
    return prev[-1]


def rouge_l(prediction: str, reference: str, *, beta: float = 1.2) -> float:
    """Sentence-level ROUGE-L F-measure over the longest common subsequence.

    beta=1.2 is the value in the original ROUGE package, which weights recall
    slightly above precision -- the right bias for reports, where omitting a
    finding matters more than adding a word.
    """
    p_tok, r_tok = tokenize(prediction), tokenize(reference)
    if not p_tok or not r_tok:
        return 0.0
    lcs = _lcs_length(p_tok, r_tok)
    if lcs == 0:
        return 0.0
    prec, rec = lcs / len(p_tok), lcs / len(r_tok)
    b2 = beta ** 2
    return ((1 + b2) * prec * rec) / (rec + b2 * prec)


def rouge_l_corpus(predictions: list, references: list) -> float:
    if not predictions:
        return 0.0
    return sum(rouge_l(p, r) for p, r in zip(predictions, references)) / len(predictions)


def nlg_report(predictions: list, references: list) -> dict:
    """Everything we can compute without a model. Deliberately does NOT include
    keys for BERTScore/METEOR/GREEN -- an absent metric should be visibly absent
    rather than silently zero."""
    out = bleu_n(predictions, references)
    out["ROUGE-L"] = rouge_l_corpus(predictions, references)
    out["n"] = len(predictions)
    return out
