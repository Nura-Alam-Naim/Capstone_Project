"""
CSV Data Loader for Adaptive-HTFL
==================================
Loads MATLAB-generated CSV datasets from the csv_data/ directory.

The CSV data has the same schema as the synthetic generator:
  - 10 sensor features (temperature_c, humidity_pct, co2_ppm, etc.)
  - 10 activity classes (0–9)
  - Hardware context per client

Data is loaded as raw (un-normalized) NumPy arrays so that the existing
normalize_features() → create_non_iid_partitions() pipeline can be reused.
"""

import os
import csv
import numpy as np
from typing import Tuple, Dict

# Default directory relative to this file
_DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csv_data")

SENSOR_NAMES = [
    "temperature_c", "humidity_pct", "co2_ppm", "motion_score",
    "light_lux", "air_quality_idx", "sound_db", "occupancy_est",
    "power_kw", "vibration_mg",
]

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

HARDWARE_NAMES = [
    "battery_pct", "signal_strength_dbm", "cpu_load_pct",
    "uptime_hrs", "packet_loss_pct",
]


def _read_csv(filepath):
    """Read a CSV file and return header + rows as lists of strings."""
    with open(filepath, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    return header, rows


def load_csv_dataset(
    data_dir: str = None,
    use_raw: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the full IoT sensor dataset from CSV.

    Args:
        data_dir: Path to the CSV data directory. Defaults to data/csv_data/.
        use_raw:  If True, loads iot_sensor_data_raw.csv (un-normalized).
                  If False, loads iot_sensor_data.csv (pre-normalized).

    Returns:
        X: np.ndarray of shape (n_samples, 10) — sensor features
        y: np.ndarray of shape (n_samples,) — integer activity labels
    """
    data_dir = data_dir or _DEFAULT_DATA_DIR
    filename = "iot_sensor_data_raw.csv" if use_raw else "iot_sensor_data.csv"
    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"CSV dataset not found at: {filepath}\n"
            f"Make sure the CSV data files are placed in: {data_dir}"
        )

    header, rows = _read_csv(filepath)

    # Parse features (first 10 columns) and labels (last column)
    n_features = len(SENSOR_NAMES)
    X = np.array([[float(row[i]) for i in range(n_features)] for row in rows])
    y = np.array([int(row[n_features]) for row in rows])

    return X, y


def load_csv_hardware_context(
    data_dir: str = None,
) -> Dict[int, Dict[str, float]]:
    """
    Load hardware context metadata for each client from CSV.

    Returns:
        Dict mapping client_id → {hardware_metric: value}
        Only contains profiles for honest clients (from CSV).
    """
    data_dir = data_dir or _DEFAULT_DATA_DIR
    filepath = os.path.join(data_dir, "hardware_context.csv")

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Hardware context CSV not found at: {filepath}\n"
            f"Make sure the CSV data files are placed in: {data_dir}"
        )

    header, rows = _read_csv(filepath)

    # header: client_id, battery_pct, signal_strength_dbm, cpu_load_pct, uptime_hrs, packet_loss_pct
    contexts = {}
    for row in rows:
        cid = int(row[0])
        ctx = {}
        for i, name in enumerate(HARDWARE_NAMES):
            col_idx = header.index(name) if name in header else i + 1
            ctx[name] = float(row[col_idx])
        contexts[cid] = ctx

    return contexts


def get_csv_dataset_info() -> Dict:
    """Return dataset metadata matching get_dataset_info() format."""
    return {
        "name": "MATLAB-Generated Smart Campus IoT Sensor Data (CSV)",
        "n_features": len(SENSOR_NAMES),
        "n_classes": len(ACTIVITY_CLASSES),
        "feature_names": SENSOR_NAMES,
        "hardware_features": HARDWARE_NAMES,
        "class_names": ACTIVITY_CLASSES,
    }
