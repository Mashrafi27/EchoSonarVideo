"""BERTScore against PubMedBERT, as EchoSonar-R's Table 3 specifies
("BERTScore is computed using PubMedBERT [15]").

Kept out of nlg.py so importing the cheap stdlib metrics never requires
`bert_score`/`torch`/downloading a model -- this module is the one caller that
needs all three.
"""
PUBMEDBERT = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"


def bertscore_corpus(predictions: list, references: list, *,
                      model_type: str = PUBMEDBERT, batch_size: int = 32,
                      device: str = None) -> dict:
    """Corpus-mean precision/recall/F1. F1 is the number Table 3 reports."""
    if not predictions:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n": 0}
    from bert_score import score as _bert_score
    P, R, F1 = _bert_score(predictions, references, model_type=model_type,
                            batch_size=batch_size, device=device,
                            lang="en", verbose=False)
    return {"precision": P.mean().item(), "recall": R.mean().item(),
            "f1": F1.mean().item(), "n": len(predictions)}
