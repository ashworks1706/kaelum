import logging
from typing import List, Dict, Optional

from ..reasoning import LLMClient, Message
from .learned_verifier import LearnedVerifier

logger = logging.getLogger("kaelum.verification")


class VerificationEngine:

    def __init__(
        self,
        llm_client,
        learned_model_path: Optional[str] = None,
        pass_label_substring: str = "POSITIVE",
        fail_closed: bool = False,
        debug: bool = False,
        # Kept for call-site compatibility; no longer used internally
        use_symbolic: bool = True,
        use_factual: bool = False,
        embedding_model: str = "all-MiniLM-L6-v2",
        use_learned_only: bool = True,
    ):
        self.llm_client = llm_client
        self.debug = debug
        self.fail_closed = fail_closed
        self.learned_verifier = None
        if learned_model_path:
            self.learned_verifier = LearnedVerifier(
                learned_model_path, label_pass_substring=pass_label_substring
            )

    def verify(
        self,
        query: str,
        reasoning_steps: List[str],
        answer: str,
        worker_type: Optional[str] = None,
    ) -> dict:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"VERIFICATION: Starting verification for {(worker_type or 'unknown').upper()} worker")
        logger.info(f"  Query: {query[:100]}...")
        logger.info(f"  Steps: {len(reasoning_steps)} reasoning steps")

        if not self.learned_verifier:
            if self.fail_closed:
                raise RuntimeError("Learned verifier is required but not configured.")
            logger.warning("Learned verifier not configured; verification failing closed.")
            return {"passed": False, "confidence": 0.0, "issues": ["Verifier unavailable"], "details": {}}

        learned_result = self.learned_verifier.score(query, answer, reasoning_steps, worker_type)
        passed = learned_result.get("passed", False)
        confidence = learned_result.get("confidence", 0.0)
        return {
            "passed": passed,
            "confidence": confidence,
            "issues": [] if passed else ["Learned verifier rejected output"],
            "details": {"label": learned_result.get("label", "")},
        }
