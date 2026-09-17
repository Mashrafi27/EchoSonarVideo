"""BERTScore against PubMedBERT, as EchoSonar-R's Table 3 specifies
("BERTScore is computed using PubMedBERT [15]").

Call parameters mirror Darya's actual harness
(report_generation/evaluation/report_pipeline/evaluate_reports.py): PubMedBERT
with num_layers=12 (bert_score has no registry entry for this model and raises
KeyError without it), inputs pre-truncated to 500 tokens to fit the 512-token
limit, rescale_with_baseline=True (no baseline file ships for PubMedBERT, so
bert_score skips the rescale with a warning -- kept anyway so the call matches
hers exactly). One real difference remains: she scores per report SECTION and
we score whole reports; see her evaluate_reports.py before comparing numbers.

Kept out of nlg.py so importing the cheap stdlib metrics never requires
`bert_score`/`torch`/downloading a model -- this module is the one caller that
needs all three.
"""
PUBMEDBERT = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"


def _truncate(texts, tokenizer, max_tokens=500):
    out = []
    for t in texts:
        ids = tokenizer.encode(t, add_special_tokens=True, truncation=True,
                               max_length=max_tokens)
        out.append(tokenizer.decode(ids, skip_special_tokens=True))
    return out


def bertscore_corpus(predictions: list, references: list, *,
                      model_type: str = PUBMEDBERT, batch_size: int = 64,
                      device: str = None) -> dict:
    """Corpus-mean precision/recall/F1. F1 is the number Table 3 reports."""
    if not predictions:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n": 0}
    from bert_score import score as _bert_score
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_type)
    P, R, F1 = _bert_score(_truncate(predictions, tok), _truncate(references, tok),
                            model_type=model_type, num_layers=12,
                            batch_size=batch_size, device=device,
                            lang="en", verbose=False,
                            rescale_with_baseline=True)
    return {"precision": P.mean().item(), "recall": R.mean().item(),
            "f1": F1.mean().item(), "n": len(predictions)}
