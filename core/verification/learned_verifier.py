"""Pluggable learned verifier using a Hugging Face text-classification pipeline.

This is a lightweight adapter so verification can rely on a model instead of
rule-based thresholds. The model is expected to output a label indicating pass
vs fail (default: label containing 'PASS' or 'OK'). Confidence is taken from
the classifier score.
"""

from typing import List, Optional
import logging

try:
    from transformers import pipeline
except ImportError:  # pragma: no cover
    pipeline = None


class LearnedVerifier:
    def __init__(self, model_name_or_path: str, label_pass_substring: str = "PASS", device: int = -1):
        if pipeline is None:
            raise ImportError("transformers is required for LearnedVerifier but is not installed.")
        self.logger = logging.getLogger("kaelum.learned_verifier")
        self.model_name_or_path = model_name_or_path
        self.label_pass_substring = label_pass_substring.lower()
        self.pipe = pipeline("text-classification", model=model_name_or_path, tokenizer=model_name_or_path, device=device)

    def score(self, query: str, answer: str, reasoning_steps: Optional[List[str]] = None, worker_type: Optional[str] = None) -> dict:
        """Return {'passed': bool, 'confidence': float, 'label': str} based on model output."""
        text = f"Query: {query}\nAnswer: {answer}\n"
        if reasoning_steps:
            text += "Reasoning:\n" + "\n".join(reasoning_steps)
        result = self.pipe(text, truncation=True)[0]
        label = result.get("label", "").lower()
        score = float(result.get("score", 0.0))
        passed = self.label_pass_substring in label
        self.logger.debug(f"LEARNED VERIFIER: label={label}, score={score:.3f}, passed={passed}")
        return {"passed": passed, "confidence": score, "label": label}
