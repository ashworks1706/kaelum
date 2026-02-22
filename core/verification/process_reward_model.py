"""Learned Process Reward Model (PRM) for LATS node scoring.

Trains a small MLP regression head on top of sentence-transformer embeddings
to score (query, reasoning_step) pairs. Replaces/augments the hand-coded
heuristics in RewardModel with a signal learned from actual verification
outcomes and human feedback ratings.

Architecture:
    features = concat(query_emb[384], step_emb[384], context_emb[384], worker_onehot[6])
    MLP: 1158 → 256 → 64 → 1  (ReLU + Dropout, sigmoid output)

Training signal (priority order):
    1. human_score  — explicit rating from human_feedback.py  [0, 1]
    2. verification_passed — binary outcome from VerificationEngine  {0, 1}

Training schedule:
    - Activates automatically when MIN_SAMPLES examples are available
    - Re-triggers every RETRAIN_INTERVAL new samples thereafter
    - Weights and data stored at ~/.kaelum/prm/

Prediction:
    blend_alpha = min(1.0, n_samples / TARGET_SAMPLES)
    final_reward = blend_alpha * prm_score + (1 - blend_alpha) * heuristic_score

    blend_alpha scales from 0 → 1 as training data accumulates, so the model
    starts fully heuristic and gradually shifts to learned rewards.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger("kaelum.prm")

# ── hyperparameters ────────────────────────────────────────────────────────────
MIN_SAMPLES = 0           # activate immediately (no heuristic fallback)
TARGET_SAMPLES = 200      # sample count at which blend_alpha reaches 1.0
RETRAIN_INTERVAL = 25     # retrain every N *new* samples
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.15

WORKER_TYPES = ["math", "code", "logic", "factual", "creative", "analysis"]
EMB_DIM = 384             # all-MiniLM-L6-v2 output size
INPUT_DIM = EMB_DIM * 3 + len(WORKER_TYPES)   # 1158


# ── singleton ──────────────────────────────────────────────────────────────────
_instance: Optional["ProcessRewardModel"] = None


def get_prm(embedding_model: str = "all-MiniLM-L6-v2") -> "ProcessRewardModel":
    """Return module-level singleton ProcessRewardModel."""
    global _instance
    if _instance is None:
        _instance = ProcessRewardModel(embedding_model=embedding_model)
    return _instance


# ── model class ───────────────────────────────────────────────────────────────
class ProcessRewardModel:
    """Trainable step-level reward model with automatic online learning."""

    def __init__(
        self,
        data_dir: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.data_dir = Path(data_dir or Path.home() / ".kaelum" / "prm")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.data_path = self.data_dir / "training_data.jsonl"
        self.weights_path = self.data_dir / "prm_weights.pt"

        from core.shared_encoder import get_shared_encoder
        self._encoder = get_shared_encoder(embedding_model, device="cpu")

        self._model = None          # PyTorch MLP, lazy-initialised
        self._training_data: list = []
        self._n_since_last_train = 0
        self.blend_alpha = 0.0      # fraction of PRM vs heuristic in final reward

        self._load_data()
        self._maybe_load_weights()
        if self._model is None:
            self._model = self._build_model()
            self._model.eval()

    # ── internal helpers ──────────────────────────────────────────────────────

    def _build_model(self):
        import torch.nn as nn

        class _MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(INPUT_DIM, 256),
                    nn.ReLU(),
                    nn.Dropout(DROPOUT),
                    nn.Linear(256, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                    nn.Sigmoid(),
                )

            def forward(self, x):
                return self.net(x).squeeze(-1)

        return _MLP()

    def _featurize(
        self,
        query: str,
        step: str,
        context_steps: List[str],
        worker_type: str,
    ) -> np.ndarray:
        """Build a 1158-dim feature vector for one (query, step) pair."""
        q_emb = self._encoder.encode(query, show_progress_bar=False).astype(np.float32)
        s_emb = self._encoder.encode(step, show_progress_bar=False).astype(np.float32)

        if context_steps:
            ctx_text = " ".join(context_steps[-3:])
            c_emb = self._encoder.encode(ctx_text, show_progress_bar=False).astype(np.float32)
        else:
            c_emb = np.zeros(EMB_DIM, dtype=np.float32)

        wt = worker_type.lower() if worker_type else "logic"
        w_idx = WORKER_TYPES.index(wt) if wt in WORKER_TYPES else 2
        w_hot = np.zeros(len(WORKER_TYPES), dtype=np.float32)
        w_hot[w_idx] = 1.0

        return np.concatenate([q_emb, s_emb, c_emb, w_hot])

    def _load_data(self) -> None:
        if not self.data_path.exists():
            return
        with open(self.data_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self._training_data.append(json.loads(line))
        n = len(self._training_data)
        self.blend_alpha = min(1.0, n / TARGET_SAMPLES)
        logger.info(f"PRM: Loaded {n} training examples (blend_alpha={self.blend_alpha:.2f})")

    def _maybe_load_weights(self) -> None:
        if self.weights_path.exists():
            import torch
            self._model = self._build_model()
            self._model.load_state_dict(
                torch.load(self.weights_path, map_location="cpu")
            )
            self._model.eval()
            logger.info("PRM: Loaded pre-trained weights")
        else:
            if len(self._training_data) >= MIN_SAMPLES:
                self._train()

    # ── training ──────────────────────────────────────────────────────────────

    def _train(self) -> None:
        import torch
        import torch.nn as nn
        from torch.optim import Adam

        data = self._training_data

        logger.info(f"PRM: Training on {len(data)} examples...")
        t0 = time.time()

        features, labels = [], []
        for ex in data:
            feat = self._featurize(
                ex["query"], ex["step"],
                ex.get("context", []), ex.get("worker_type", "logic")
            )
            features.append(feat)
            labels.append(float(ex["label"]))

        if len(features) == 0:
            return

        X = torch.tensor(np.array(features), dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.float32)

        if self._model is None:
            self._model = self._build_model()

        optimizer = Adam(self._model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        loss_fn = nn.BCELoss()

        self._model.train()
        last_loss = 0.0
        for _ in range(EPOCHS):
            optimizer.zero_grad()
            preds = self._model(X)
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()
            last_loss = loss.item()

        self._model.eval()
        self._n_since_last_train = 0

        torch.save(self._model.state_dict(), self.weights_path)
        logger.info(
            f"PRM: Training complete in {time.time()-t0:.1f}s "
            f"(loss={last_loss:.4f}, samples={len(features)}) → {self.weights_path}"
        )

    # ── public API ────────────────────────────────────────────────────────────

    def record(
        self,
        query: str,
        step: str,
        context_steps: List[str],
        worker_type: str,
        verification_passed: bool,
        human_score: Optional[float] = None,
    ) -> None:
        """Record one training example.

        human_score takes priority over verification_passed when provided.
        """
        label = human_score if human_score is not None else float(verification_passed)
        example = {
            "query": query,
            "step": step,
            "context": (context_steps or [])[-3:],
            "worker_type": worker_type,
            "label": label,
            "ts": time.time(),
        }
        self._training_data.append(example)
        self._n_since_last_train += 1

        with open(self.data_path, "a") as f:
            f.write(json.dumps(example) + "\n")

        n = len(self._training_data)
        self.blend_alpha = min(1.0, n / TARGET_SAMPLES)

        if n >= MIN_SAMPLES and self._n_since_last_train >= RETRAIN_INTERVAL:
            self._train()

    def predict_step_quality(
        self,
        query: str,
        step: str,
        context_steps: Optional[List[str]] = None,
        worker_type: str = "logic",
    ) -> Optional[float]:
        """Predict step quality ∈ [0, 1]. Always uses the PRM model."""
        if self._model is None:
            self._model = self._build_model()
            self._model.eval()
        import torch
        feat = self._featurize(query, step, context_steps or [], worker_type)
        x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return float(self._model(x).item())

    def blend(self, heuristic_score: float, prm_score: float) -> float:
        """Blend heuristic and PRM scores according to current blend_alpha."""
        return (1.0 - self.blend_alpha) * heuristic_score + self.blend_alpha * prm_score

    @property
    def is_active(self) -> bool:
        """True when the PRM is available."""
        return self._model is not None

    @property
    def n_samples(self) -> int:
        return len(self._training_data)
