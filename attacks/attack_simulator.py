"""
Attack Simulation Module — Adaptive-HTFL
Label flip, noise injection, scaling, slow poison.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional


ATTACK_REGISTRY = {
    "none":        {"name": "No Attack (Baseline)",       "severity": "none"},
    "label_flip":  {"name": "Label Flipping Attack",      "severity": "high"},
    "noise":       {"name": "Noise Injection Attack",     "severity": "high"},
    "scaling":     {"name": "Scaling / Byzantine Attack", "severity": "very_high"},
    "slow_poison": {"name": "Slow Poisoning Attack",      "severity": "medium"},
}


def get_attack_config(
    attack_type: str,
    malicious_fraction: float = 0.30,
    n_clients: int = 10,
    random_seed: int = 42,
) -> Tuple[List[int], List[Optional[str]], List[Optional[Dict]]]:
    np.random.seed(random_seed)
    n_mal = max(1, int(n_clients * malicious_fraction)) if attack_type != "none" else 0
    mal_ids = np.random.choice(n_clients, n_mal, replace=False).tolist() if n_mal > 0 else []

    attack_types, attack_params = [], []
    for cid in range(n_clients):
        if cid in mal_ids:
            attack_types.append(attack_type)
            attack_params.append(_default_params(attack_type))
        else:
            attack_types.append(None)
            attack_params.append(None)

    return mal_ids, attack_types, attack_params


def _default_params(attack_type: str) -> Dict:
    return {
        "label_flip":  {"target_class": 0, "flip_to": 1},
        "noise":       {"noise_scale": 3.0},
        "scaling":     {"scale_factor": 10.0},
        "slow_poison": {"poison_rate": 0.15},
    }.get(attack_type, {})


def get_experiment_scenarios() -> List[Dict]:
    return [
        {"id": "baseline",    "attack": "none",        "malicious_fraction": 0.00, "label": "Baseline (No Attack)",       "color": "#1D9E75"},
        {"id": "label_flip",  "attack": "label_flip",  "malicious_fraction": 0.30, "label": "Label Flip (30% malicious)", "color": "#E24B4A"},
        {"id": "noise",       "attack": "noise",       "malicious_fraction": 0.30, "label": "Noise Injection (30%)",      "color": "#D4537E"},
        {"id": "scaling",     "attack": "scaling",     "malicious_fraction": 0.20, "label": "Scaling Attack (20%)",       "color": "#7F77DD"},
        {"id": "slow_poison", "attack": "slow_poison", "malicious_fraction": 0.40, "label": "Slow Poisoning (40%)",       "color": "#BA7517"},
    ]
