"""
Synthetic Smart Campus IoT Data Generator
Generates sensor readings + hardware context metadata per client.

Sensor modalities (10):
  temperature_c, humidity_pct, co2_ppm, motion_score,
  light_lux, air_quality_idx, sound_db, occupancy_est,
  power_kw, vibration_mg

Hardware context per client (for Contextual/Hardware Trust):
  battery_pct, signal_strength_dbm, cpu_load_pct, uptime_hrs, packet_loss_pct
"""

import numpy as np
from typing import Tuple, List, Dict, Optional


ACTIVITY_CLASSES = [
    "Empty Room",
    "Single Occupant Working",
    "Small Meeting (2-5)",
    "Large Meeting (6-15)",
    "Lecture Class",
    "Lab Session",
    "Cafeteria Activity",
    "Gym/Sports",
    "Corridor Traffic",
    "Emergency/Alarm",
]

SENSOR_NAMES = [
    "temperature_c", "humidity_pct", "co2_ppm", "motion_score",
    "light_lux", "air_quality_idx", "sound_db", "occupancy_est",
    "power_kw", "vibration_mg",
]

HARDWARE_NAMES = [
    "battery_pct", "signal_strength_dbm", "cpu_load_pct",
    "uptime_hrs", "packet_loss_pct",
]

# Realistic sensor profiles per class [mean, std]
CLASS_PROFILES = {
    0: [[20.0,0.5],[40.0,3.0],[400.0,20.0],[0.0,0.1],[100.0,30.0],[95.0,2.0],[30.0,3.0],[0.0,0.5],[0.5,0.1],[0.1,0.05]],
    1: [[21.5,0.8],[45.0,4.0],[550.0,40.0],[1.5,0.5],[300.0,50.0],[92.0,3.0],[38.0,5.0],[1.2,0.3],[1.2,0.3],[0.3,0.1]],
    2: [[22.0,1.0],[48.0,4.0],[750.0,80.0],[3.5,0.8],[350.0,60.0],[88.0,4.0],[48.0,6.0],[3.5,0.8],[1.8,0.4],[0.5,0.15]],
    3: [[23.0,1.2],[52.0,5.0],[1100.0,120.0],[6.0,1.2],[400.0,70.0],[82.0,5.0],[58.0,8.0],[9.0,1.5],[3.0,0.6],[0.9,0.2]],
    4: [[23.5,1.5],[55.0,5.0],[1500.0,150.0],[4.0,1.0],[450.0,80.0],[78.0,6.0],[55.0,7.0],[28.0,4.0],[5.0,0.8],[1.2,0.3]],
    5: [[22.5,1.2],[50.0,4.5],[900.0,100.0],[5.0,1.1],[500.0,90.0],[85.0,5.0],[52.0,7.0],[15.0,3.0],[4.0,0.7],[2.5,0.5]],
    6: [[24.0,1.5],[58.0,6.0],[1200.0,130.0],[8.0,1.5],[380.0,65.0],[75.0,7.0],[72.0,10.0],[40.0,8.0],[8.0,1.2],[1.8,0.4]],
    7: [[25.0,2.0],[65.0,8.0],[800.0,90.0],[9.5,1.8],[600.0,100.0],[70.0,8.0],[80.0,12.0],[20.0,5.0],[6.0,1.0],[5.0,1.0]],
    8: [[21.0,1.0],[43.0,3.5],[480.0,50.0],[7.0,1.5],[250.0,40.0],[90.0,3.0],[55.0,8.0],[5.0,1.5],[1.0,0.2],[1.5,0.3]],
    9: [[22.0,3.0],[50.0,8.0],[900.0,200.0],[10.0,2.0],[800.0,150.0],[60.0,15.0],[95.0,15.0],[15.0,5.0],[12.0,3.0],[8.0,2.0]],
}


def generate_iot_dataset(
    n_samples: int = 7500,
    n_features: int = 10,
    n_classes: int = 10,
    noise_level: float = 0.05,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic IoT sensor dataset."""
    if random_seed is not None:
        np.random.seed(random_seed)
    X_list, y_list = [], []
    samples_per_class = n_samples // n_classes

    for cls in range(n_classes):
        profile = CLASS_PROFILES[cls]
        n = samples_per_class + (n_samples % n_classes if cls == 0 else 0)
        features = np.column_stack([
            np.random.normal(profile[f][0], profile[f][1], n)
            for f in range(n_features)
        ])
        features += np.random.normal(0, noise_level, features.shape) * features
        X_list.append(features)
        y_list.append(np.full(n, cls))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    idx = np.random.permutation(len(X))
    return X[idx], y[idx].astype(int)


def normalize_features(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    return (X - X_min) / (X_max - X_min + 1e-8), X_min, X_max


def generate_hardware_context(
    n_clients: int,
    malicious_ids: List[int] = None,
    faulty_ids: List[int] = None,
    random_seed: Optional[int] = None,
) -> Dict[int, Dict[str, float]]:
    """
    Generate realistic hardware context metadata per client.
    Malicious/faulty clients have degraded hardware signals.

    Returns dict: {client_id: {hardware_metric: value}}
    """
    np.random.seed(random_seed)
    malicious_ids = malicious_ids or []
    faulty_ids = faulty_ids or []
    contexts = {}

    for cid in range(n_clients):
        if cid in malicious_ids:
            # Malicious nodes may fake good hardware but have subtle anomalies
            ctx = {
                "battery_pct": float(np.random.uniform(60, 95)),
                "signal_strength_dbm": float(np.random.uniform(-75, -50)),
                "cpu_load_pct": float(np.random.uniform(70, 95)),  # suspiciously high
                "uptime_hrs": float(np.random.uniform(1, 48)),     # frequent restarts
                "packet_loss_pct": float(np.random.uniform(5, 20)),
            }
        elif cid in faulty_ids:
            # Faulty hardware — genuinely degraded
            ctx = {
                "battery_pct": float(np.random.uniform(5, 30)),
                "signal_strength_dbm": float(np.random.uniform(-95, -80)),
                "cpu_load_pct": float(np.random.uniform(85, 99)),
                "uptime_hrs": float(np.random.uniform(500, 2000)),
                "packet_loss_pct": float(np.random.uniform(15, 40)),
            }
        else:
            # Healthy honest nodes
            ctx = {
                "battery_pct": float(np.random.uniform(70, 100)),
                "signal_strength_dbm": float(np.random.uniform(-65, -40)),
                "cpu_load_pct": float(np.random.uniform(10, 55)),
                "uptime_hrs": float(np.random.uniform(100, 800)),
                "packet_loss_pct": float(np.random.uniform(0, 3)),
            }
        contexts[cid] = ctx
    return contexts


def create_non_iid_partitions(
    X: np.ndarray,
    y: np.ndarray,
    n_clients: int,
    alpha: float = 0.5,
    random_seed: Optional[int] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Non-IID partitions via Dirichlet distribution."""
    if random_seed is not None:
        np.random.seed(random_seed)
    n_classes = len(np.unique(y))
    client_data = [[] for _ in range(n_clients)]
    client_labels = [[] for _ in range(n_clients)]

    for cls in range(n_classes):
        cls_idx = np.where(y == cls)[0]
        np.random.shuffle(cls_idx)
        proportions = np.random.dirichlet(alpha * np.ones(n_clients))
        proportions = (proportions * len(cls_idx)).astype(int)
        proportions[-1] = len(cls_idx) - proportions[:-1].sum()
        start = 0
        for c, count in enumerate(proportions):
            end = start + count
            client_data[c].append(X[cls_idx[start:end]])
            client_labels[c].append(y[cls_idx[start:end]])
            start = end

    partitions = []
    for c in range(n_clients):
        Xc = np.vstack([d for d in client_data[c] if len(d) > 0])
        yc = np.concatenate([d for d in client_labels[c] if len(d) > 0])
        idx = np.random.permutation(len(Xc))
        partitions.append((Xc[idx], yc[idx]))
    return partitions


def get_dataset_info() -> Dict:
    return {
        "name": "Synthetic Smart Campus IoT Sensor Data",
        "n_features": 10,
        "n_classes": 10,
        "feature_names": SENSOR_NAMES,
        "hardware_features": HARDWARE_NAMES,
        "class_names": ACTIVITY_CLASSES,
    }
