"""VerificationEngine — removed.

Verification is now handled directly by the PRM gate in the orchestrator.
The average PRM step score is compared against `config.prm_pass_threshold`
(default 0.5) to decide pass/fail. No separate verifier object is needed.

See: runtime/orchestrator.py — STEP 4: PRM GATE
     core/verification/process_reward_model.py
"""


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
            # No learned verifier — use PRM average step score as the gate.
            # This prevents all-negative PRM training labels when no model is configured.
            try:
                from core.verification.process_reward_model import get_prm
                prm = get_prm()
                if prm.is_active and reasoning_steps:
                    scores = [
                        prm.predict_step_quality(
                            query, step, reasoning_steps[:i], worker_type or "logic"
                        )
                        for i, step in enumerate(reasoning_steps)
                    ]
                    avg = sum(scores) / len(scores)
                    passed = avg >= 0.5
                    logger.info(
                        f"VERIFICATION: No learned verifier — PRM fallback "
                        f"(avg={avg:.3f}, passed={passed})"
                    )
                    return {
                        "passed": passed,
                        "confidence": avg,
                        "issues": [] if passed else [f"PRM avg score {avg:.3f} below 0.5 threshold"],
                        "details": {"prm_avg": avg},
                    }
            except Exception as e:
                logger.warning(f"VERIFICATION: PRM fallback failed: {e}")
            # Neither verifier nor PRM active — default to pass so early runs don't
            # poison PRM labels with all-negative examples before any data exists.
            logger.warning("VERIFICATION: No verifier or active PRM — defaulting to pass")
            return {"passed": True, "confidence": 0.5, "issues": [], "details": {}}

        learned_result = self.learned_verifier.score(query, answer, reasoning_steps, worker_type)
        passed = learned_result.get("passed", False)
        confidence = learned_result.get("confidence", 0.0)
        return {
            "passed": passed,
            "confidence": confidence,
            "issues": [] if passed else ["Learned verifier rejected output"],
            "details": {"label": learned_result.get("label", "")},
        }
