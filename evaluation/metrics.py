"""
Evaluation Metrics — Adaptive-HTFL
Tracks accuracy, detection rate, false positive rate,
sparsification efficiency, and DPoT consensus stats.
"""

import numpy as np
import json
import os
from typing import List, Dict, Optional


class ExperimentMetrics:

    def __init__(self, scenario_id: str, strategy: str, n_rounds: int):
        self.scenario_id = scenario_id
        self.strategy = strategy
        self.n_rounds = n_rounds
        self.accuracy_history: List[float] = []
        self.loss_history: List[float] = []
        self.trust_history: List[List[float]] = []
        self.flagged_history: List[List[int]] = []
        self.compression_history: List[float] = []
        self.dpot_history: List[Optional[Dict]] = []
        self.malicious_ids: List[int] = []

    def log_round(self, accuracy, loss, trust_scores=None,
                  flagged=None, compression=0.0, dpot=None):
        self.accuracy_history.append(float(accuracy))
        self.loss_history.append(float(loss))
        if trust_scores:
            self.trust_history.append([float(t) for t in trust_scores])
        if flagged is not None:
            self.flagged_history.append(flagged)
        self.compression_history.append(float(compression))
        self.dpot_history.append(dpot)

    def final_accuracy(self) -> float:
        return float(np.mean(self.accuracy_history[-3:])) if self.accuracy_history else 0.0

    def convergence_round(self, threshold=0.75) -> int:
        for i, a in enumerate(self.accuracy_history):
            if a >= threshold:
                return i + 1
        return self.n_rounds

    def detection_rate(self) -> float:
        if not self.flagged_history or not self.malicious_ids:
            return 0.0
        flagged = set(self.flagged_history[-1])
        mal = set(self.malicious_ids)
        return len(flagged & mal) / len(mal) if mal else 1.0

    def false_positive_rate(self, n_clients=10) -> float:
        if not self.flagged_history:
            return 0.0
        flagged = set(self.flagged_history[-1])
        mal = set(self.malicious_ids)
        honest = set(range(n_clients)) - mal
        fp = flagged - mal
        return len(fp) / len(honest) if honest else 0.0

    def avg_compression(self) -> float:
        return float(np.mean(self.compression_history)) if self.compression_history else 0.0

    def dpot_consensus_rate(self) -> float:
        dpot_rounds = [d for d in self.dpot_history if d is not None]
        if not dpot_rounds:
            return 0.0
        return sum(1 for d in dpot_rounds if d.get("consensus", False)) / len(dpot_rounds)

    def to_dict(self) -> Dict:
        return {
            "scenario_id": self.scenario_id,
            "strategy": self.strategy,
            "n_rounds": self.n_rounds,
            "final_accuracy": self.final_accuracy(),
            "accuracy_history": self.accuracy_history,
            "loss_history": self.loss_history,
            "convergence_round": self.convergence_round(),
            "detection_rate": self.detection_rate(),
            "false_positive_rate": self.false_positive_rate(),
            "avg_compression": self.avg_compression(),
            "dpot_consensus_rate": self.dpot_consensus_rate(),
            "trust_history": self.trust_history,
            "dpot_history": [d for d in self.dpot_history if d],
            "malicious_ids": self.malicious_ids,
        }


class ResultsManager:

    def __init__(self, results_dir="results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    def save(self, results: List[Dict], filename="experiment_results.json"):
        path = os.path.join(self.results_dir, filename)
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved: {path}")
        return path

    def load(self, filename="experiment_results.json") -> List[Dict]:
        path = os.path.join(self.results_dir, filename)
        if not os.path.exists(path):
            return []
        with open(path) as f:
            return json.load(f)

    def build_comparison_table(self, results: List[Dict]) -> List[Dict]:
        scenarios = {}
        for r in results:
            sid, strat = r["scenario_id"], r["strategy"]
            if sid not in scenarios:
                scenarios[sid] = {}
            scenarios[sid][strat] = r

        table = []
        for sid, strats in scenarios.items():
            fa  = strats.get("FedAvg",        {}).get("final_accuracy", 0)
            bt  = strats.get("BasicTrust",    {}).get("final_accuracy", 0)
            htfl = strats.get("AdaptiveHTFL", {}).get("final_accuracy", 0)
            dr  = strats.get("AdaptiveHTFL",  {}).get("detection_rate", 0)
            fpr = strats.get("AdaptiveHTFL",  {}).get("false_positive_rate", 0)
            comp = strats.get("AdaptiveHTFL", {}).get("avg_compression", 0)
            dpot = strats.get("AdaptiveHTFL", {}).get("dpot_consensus_rate", 0)
            table.append({
                "scenario": sid,
                "fedavg": fa,
                "basic_trust": bt,
                "adaptive_htfl": htfl,
                "improvement_over_fedavg": htfl - fa,
                "detection_rate": dr,
                "false_positive_rate": fpr,
                "avg_compression": comp,
                "dpot_consensus_rate": dpot,
            })
        return table
