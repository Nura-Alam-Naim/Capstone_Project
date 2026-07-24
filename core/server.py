"""
Central Aggregation Server — Adaptive-HTFL
Supports three strategies for comparison:
  1. FedAvg       — baseline, no trust
  2. BasicTrust   — simple cosine similarity trust (prior work benchmark)
  3. AdaptiveHTFL — full framework (Multi-Dim Trust + Sparsification + DPoT)
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from core.client import LocalModel
from core.trust_engine_ai import MultiDimensionalTrustEngine
from blockchain.dpot_chain import DPoTChain, SmartContract


class AdaptiveHTFLServer:

    def __init__(
        self,
        n_features: int = 10,
        n_classes: int = 10,
        n_clients: int = 10,
        strategy: str = "adaptive_htfl",   # "fedavg" | "basic_trust" | "adaptive_htfl"
        trust_config: Optional[Dict] = None,
        use_dpot: bool = True,
    ):
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_clients = n_clients
        self.strategy = strategy
        self.use_dpot = use_dpot and (strategy == "adaptive_htfl")

        self.global_model = LocalModel(n_features, n_classes)
        self.round_num = 0

        cfg = trust_config or {}

        if strategy == "adaptive_htfl":
            self.trust_engine = MultiDimensionalTrustEngine(
                n_clients=n_clients,
                alpha=cfg.get("alpha", 0.30),
                beta=cfg.get("beta", 0.45),
                gamma=cfg.get("gamma", 0.25),
                temporal_decay=cfg.get("temporal_decay", 0.80),
                pca_components=cfg.get("pca_components", 5),
                latent_trust_mode=cfg.get("latent_trust_mode", "autoencoder_anomaly"),
                latent_dim=cfg.get("latent_dim", 6),
                autoencoder_max_iter=cfg.get("autoencoder_max_iter", 300),
                warmup_rounds=cfg.get("warmup_rounds", 2),
                reference_pool_size=cfg.get("reference_pool_size", 128),
                reference_trust_threshold=cfg.get("reference_trust_threshold", 0.60),
                use_supervised_classifier=cfg.get("use_supervised_classifier", False),
                random_seed=cfg.get("random_seed", 42),
            )
        elif strategy == "basic_trust":
            # Simplified trust engine: cosine similarity only
            self.trust_engine = MultiDimensionalTrustEngine(
                n_clients=n_clients,
                alpha=0.0,   # no hardware trust
                beta=1.0,    # only latent-space trust
                gamma=0.0,   # no temporal
                latent_trust_mode=cfg.get("basic_trust_mode", "pca_cosine"),
                latent_dim=cfg.get("latent_dim", 6),
                autoencoder_max_iter=cfg.get("autoencoder_max_iter", 300),
                warmup_rounds=cfg.get("warmup_rounds", 2),
                random_seed=cfg.get("random_seed", 42),
            )
        else:
            self.trust_engine = None

        # DPoT chain
        if self.use_dpot:
            self.dpot = DPoTChain(contract=SmartContract(
                base_threshold=cfg.get("dpot_threshold", 0.55),
                min_committee_size=cfg.get("min_committee", 3),
            ))
        else:
            self.dpot = None

        self.global_accuracy_history: List[float] = []
        self.trust_score_history: List[np.ndarray] = []
        self.dpot_log: List[Dict] = []
        self.compression_log: List[float] = []

    def get_global_weights(self) -> Dict:
        return self.global_model.get_weights()

    def get_client_trust_scores(self) -> Dict[int, float]:
        """Return latest trust scores for all clients (for sparsification)."""
        if self.trust_engine and self.trust_engine.trust_history:
            last = self.trust_engine.trust_history[-1]
            return {i: float(last[i]) for i in range(len(last))}
        return {i: 1.0 for i in range(self.n_clients)}

    def aggregate(
        self,
        client_weights: List[Dict],
        n_samples: List[int],
        client_ids: List[int],
        client_metas: List[Dict],
    ) -> Dict:
        """
        Aggregate client updates using the selected strategy.
        Returns a log dict.
        """
        self.round_num += 1

        global_flat = self.global_model.get_flat_weights()
        flat_updates = [
            np.concatenate([w["W"].flatten(), w["b"].flatten()]) - global_flat
            for w in client_weights
        ]

        hardware_contexts = [m.get("hardware_context", {}) for m in client_metas]
        avg_compression = float(np.mean([m.get("compression_ratio", 0.0) for m in client_metas]))
        self.compression_log.append(avg_compression)

        if self.strategy == "fedavg":
            n_arr = np.array(n_samples, dtype=float)
            weights_coef = n_arr / n_arr.sum()
            trust_scores = np.ones(len(client_ids))
            log = {
                "strategy": "FedAvg",
                "trust_scores": trust_scores.tolist(),
                "flagged": [],
                "avg_compression": 0.0,
                "dpot": None,
            }

        else:
            # Trust-FL or Adaptive-HTFL
            trust_labels = [m.get("trust_label") for m in client_metas]
            if not any(label is not None for label in trust_labels):
                trust_labels = None
            trust_scores = self.trust_engine.compute_trust_scores(
                flat_updates,
                client_ids,
                n_samples,
                hardware_contexts,
                trust_labels=trust_labels,
            )
            weights_coef = self.trust_engine.compute_aggregation_weights(
                trust_scores, n_samples
            )
            summary = self.trust_engine.get_summary(client_ids)
            log = {
                "strategy": self.strategy,
                "trust_scores": trust_scores.tolist(),
                "flagged": summary.get("flagged", []),
                "mean_trust": summary.get("mean_trust", 1.0),
                "avg_compression": avg_compression,
                "dpot": None,
            }

        self.trust_score_history.append(trust_scores)

        # ── Server-side Norm Clipping (robust aggregation) ──
        # Clip each client's update to limit the impact of any single client.
        # Uses median norm as reference — resilient to outliers.
        # Applied only to AdaptiveHTFL; FedAvg and BasicTrust stay as clean baselines.
        global_w = self.get_global_weights()
        if self.strategy == "adaptive_htfl":
            update_norms = []
            for w in client_weights:
                delta = np.concatenate([
                    (w["W"] - global_w["W"]).flatten(),
                    (w["b"] - global_w["b"]).flatten(),
                ])
                update_norms.append(float(np.linalg.norm(delta)))

            median_norm = float(np.median(update_norms))
            clip_threshold = median_norm * 1.2

            clipped_weights = []
            for i, w in enumerate(client_weights):
                if clip_threshold > 1e-10 and update_norms[i] > clip_threshold:
                    scale = clip_threshold / update_norms[i]
                    clipped_weights.append({
                        "W": global_w["W"] + (w["W"] - global_w["W"]) * scale,
                        "b": global_w["b"] + (w["b"] - global_w["b"]) * scale,
                    })
                else:
                    clipped_weights.append(w)
            client_weights = clipped_weights

        # Direct aggregation
        new_W = np.zeros((self.n_features, self.n_classes))
        new_b = np.zeros(self.n_classes)

        for w, coef in zip(client_weights, weights_coef):
            new_W += coef * w["W"]
            new_b += coef * w["b"]

        self.global_model.set_weights({"W": new_W, "b": new_b})

        # DPoT consensus (Adaptive-HTFL only)
        if self.use_dpot and self.dpot:
            trust_dict = {client_ids[i]: float(trust_scores[i]) for i in range(len(client_ids))}
            block = self.dpot.propose_round(
                self.round_num, self.global_model.get_weights(), trust_dict
            )
            dpot_summary = {
                "committee": block.committee,
                "excluded": block.excluded,
                "threshold": block.dynamic_threshold,
                "votes": block.consensus_votes,
                "consensus": block.consensus_reached,
                "fingerprint": block.model_fingerprint,
                "hash": block.block_hash,
            }
            log["dpot"] = dpot_summary
            self.dpot_log.append(dpot_summary)

        return log

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        acc = self.global_model.accuracy(X_test, y_test)
        self.global_accuracy_history.append(acc)
        return acc

    def get_dpot_stats(self) -> Dict:
        if not self.dpot:
            return {}
        return {
            "chain_length": len(self.dpot.chain),
            "consensus_rate": self.dpot.consensus_rate(),
            "avg_committee_size": self.dpot.average_committee_size(),
            "chain_integrity": self.dpot.verify_chain_integrity(),
        }
