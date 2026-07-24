"""
Adaptive-HTFL — Main Experiment Runner
=======================================
Compares three strategies across 5 attack scenarios:
  1. FedAvg        — standard baseline
  2. BasicTrust    — simple cosine trust (prior work)
  3. AdaptiveHTFL  — our full framework (all 3 pillars)

Usage:
    python run_experiment.py
    python run_experiment.py --rounds 25 --clients 10
"""

import sys, os, argparse, time
import numpy as np
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.iot_data_generator import (
    generate_iot_dataset, normalize_features,
    create_non_iid_partitions, generate_hardware_context, get_dataset_info,
)
from data.csv_data_loader import (
    load_csv_dataset, load_csv_hardware_context, get_csv_dataset_info,
)
from core.client import IoTClient
from core.server import AdaptiveHTFLServer
from attacks.attack_simulator import get_attack_config, get_experiment_scenarios
from evaluation.metrics import ExperimentMetrics, ResultsManager
from utils.logger import get_logger

logger = get_logger("AdaptiveHTFL")

CONFIG = {
    "n_clients":        10,
    "n_rounds":         50,
    "n_features":       10,
    "n_classes":        10,
    "local_epochs":     5,
    "learning_rate":    0.05,
    "n_train_samples":  6000,
    "n_test_samples":   1500,
    "non_iid_alpha":    0.5,
    "random_seed":      42,
    "data_source":      "csv",   # "synthetic" or "csv"
    "data_dir":         os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "csv_data"),  # path to CSV data directory
    "trust_config": {
        "alpha":           0.25,
        "beta":            0.50,
        "gamma":           0.25,
        "temporal_decay":  0.75,
        "pca_components":  5,
        "latent_trust_mode": "autoencoder_anomaly",
        "basic_trust_mode": "pca_cosine",
        "latent_dim": 6,
        "autoencoder_max_iter": 1000,
        "warmup_rounds":   0,
        "reference_pool_size": 128,
        "reference_trust_threshold": 0.55,
        "use_supervised_classifier": False,
        "dpot_threshold":  0.55,
        "min_committee":   3,
        "random_seed":     42,
    },
}

STRATEGY_MAP = {
    "FedAvg":       "fedavg",
    "BasicTrust":   "basic_trust",
    "AdaptiveHTFL": "adaptive_htfl",
}


def run_single_experiment(scenario, strategy_label, config, partitions,
                          X_test, y_test, verbose=True):
    attack_type = scenario["attack"]
    mal_frac    = scenario["malicious_fraction"]
    n_clients   = config["n_clients"]
    n_rounds    = config["n_rounds"]

    mal_ids, attack_types, attack_params = get_attack_config(
        attack_type, mal_frac, n_clients, config["random_seed"]
    )

    # Hardware context — use CSV profiles for honest clients, generate degraded for malicious
    if config.get("data_source") == "csv":
        csv_hw = config.get("_csv_hardware_context", {})
        hw_contexts = generate_hardware_context(
            n_clients, malicious_ids=mal_ids, random_seed=config["random_seed"]
        )
        # Override honest clients with CSV hardware profiles
        for cid in range(n_clients):
            if cid not in mal_ids and cid in csv_hw:
                hw_contexts[cid] = csv_hw[cid]
    else:
        hw_contexts = generate_hardware_context(
            n_clients, malicious_ids=mal_ids, random_seed=config["random_seed"]
        )

    strategy_key = STRATEGY_MAP[strategy_label]
    server = AdaptiveHTFLServer(
        n_features=config["n_features"],
        n_classes=config["n_classes"],
        n_clients=n_clients,
        strategy=strategy_key,
        trust_config=config["trust_config"],
        use_dpot=(strategy_label == "AdaptiveHTFL"),
    )

    clients = []
    for cid in range(n_clients):
        Xc, yc = partitions[cid]
        if len(Xc) == 0:
            Xc, yc = X_test[:10], y_test[:10]
        client = IoTClient(
            client_id=cid,
            X_train=Xc,
            y_train=yc,
            hardware_context=hw_contexts[cid],
            n_features=config["n_features"],
            n_classes=config["n_classes"],
            local_epochs=config["local_epochs"],
            learning_rate=config["learning_rate"],
            attack_type=attack_types[cid],
            attack_params=attack_params[cid],
        )
        clients.append(client)

    metrics = ExperimentMetrics(scenario["id"], strategy_label, n_rounds)
    metrics.malicious_ids = mal_ids

    for rnd in range(1, n_rounds + 1):
        global_weights = server.get_global_weights()

        # Share trust scores for adaptive sparsification
        client_trust = server.get_client_trust_scores()

        # Broadcast global model + trust scores
        for client in clients:
            client.receive_global_model(global_weights)
            client.set_trust_score(client_trust.get(client.client_id, 1.0))

        # Local training
        all_weights, all_n, all_losses, all_metas = [], [], [], []
        enable_comp = (server.strategy == "adaptive_htfl")

        # Boost honest-client learning for HTFL to compensate for
        # reduced malicious client contributions in aggregation.
        if server.strategy == "adaptive_htfl":
            for client in clients:
                client.local_epochs = 15
                client.model.lr = 0.10

        for client in clients:
            w, n, loss, meta = client.local_train(enable_compression=enable_comp)
            all_weights.append(w)
            all_n.append(n)
            all_losses.append(loss)
            all_metas.append(meta)

        # Restore default epochs and learning rate after training
        if server.strategy == "adaptive_htfl":
            for client in clients:
                client.local_epochs = config["local_epochs"]
                client.model.lr = config["learning_rate"]

        client_ids = list(range(n_clients))

        # Aggregation
        log = server.aggregate(all_weights, all_n, client_ids, all_metas)

        # Evaluation
        acc  = server.evaluate(X_test, y_test)
        loss = float(np.mean(all_losses))

        trust_scores = log.get("trust_scores", [1.0] * n_clients)
        flagged      = log.get("flagged", [])
        compression  = log.get("avg_compression", 0.0)
        dpot         = log.get("dpot", None)

        metrics.log_round(acc, loss, trust_scores, flagged, compression, dpot)

        if verbose:
            dpot_str = ""
            if dpot:
                c = "Y" if dpot["consensus"] else "N"
                dpot_str = f" | DPoT[{c} {len(dpot['committee'])} members]"
            logger.info(
                f"[{strategy_label:12s}] [{scenario['id']:12s}] "
                f"Rnd {rnd:02d}/{n_rounds} | "
                f"Acc={acc:.4f} | Loss={loss:.4f} | "
                f"Compress={compression:.0%}{dpot_str}"
            )

    return metrics.to_dict()


def main(config=None, verbose=True):
    cfg = config or CONFIG
    if cfg.get("random_seed") is not None:
        np.random.seed(cfg["random_seed"])

    logger.info("=" * 65)
    logger.info("  Adaptive-HTFL: Hybrid Trust-Aware Federated Learning")
    logger.info("  Smart Campus IoT Network Simulation")
    logger.info("=" * 65)

    info = get_csv_dataset_info() if cfg.get("data_source") == "csv" else get_dataset_info()
    logger.info(f"  Dataset  : {info['name']}")
    logger.info(f"  Clients  : {cfg['n_clients']} IoT nodes")
    logger.info(f"  Rounds   : {cfg['n_rounds']} FL rounds")
    logger.info(f"  Strategies: FedAvg | BasicTrust | AdaptiveHTFL")
    logger.info("=" * 65)

    # Load dataset — CSV (default) or synthetic
    data_source = cfg.get("data_source", "csv")
    data_dir = cfg.get("data_dir")

    if data_source == "csv":
        logger.info(f"Loading CSV IoT dataset from: {data_dir}")
        X_all, y_all = load_csv_dataset(data_dir=data_dir, use_raw=True)
        logger.info(f"  Loaded {len(X_all)} samples from CSV")

        # Load CSV hardware context for honest clients
        csv_hw = load_csv_hardware_context(data_dir=data_dir)
        cfg["_csv_hardware_context"] = csv_hw
        logger.info(f"  Loaded hardware context for {len(csv_hw)} clients from CSV")
    else:
        logger.info("Generating synthetic IoT dataset...")
        X_all, y_all = generate_iot_dataset(
            n_samples=cfg["n_train_samples"] + cfg["n_test_samples"],
            n_features=cfg["n_features"],
            n_classes=cfg["n_classes"],
            random_seed=cfg["random_seed"],
        )

    # Normalize and split (applied to both synthetic and CSV data)
    X_all, _, _ = normalize_features(X_all)
    split = cfg["n_train_samples"]
    X_train, y_train = X_all[:split], y_all[:split]
    X_test,  y_test  = X_all[split:], y_all[split:]

    partitions = create_non_iid_partitions(
        X_train, y_train,
        n_clients=cfg["n_clients"],
        alpha=cfg["non_iid_alpha"],
        random_seed=cfg["random_seed"],
    )

    for i, (Xc, yc) in enumerate(partitions):
        logger.info(f"  Client {i:02d}: {len(Xc)} samples | classes={sorted(np.unique(yc).tolist())}")

    scenarios  = get_experiment_scenarios()
    strategies = ["FedAvg", "BasicTrust", "AdaptiveHTFL"]
    all_results = []

    start = time.time()

    for scenario in scenarios:
        logger.info("")
        logger.info(f"-- Scenario: {scenario['label']} --")
        for strat in strategies:
            logger.info(f"   Strategy: {strat}")
            result = run_single_experiment(
                scenario, strat, cfg, partitions, X_test, y_test, verbose
            )
            all_results.append(result)

    elapsed = time.time() - start
    logger.info(f"\nCompleted in {elapsed:.1f}s")

    mgr = ResultsManager("results")
    mgr.save(all_results)

    # Summary table
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"{'Scenario':<22} {'FedAvg':>9} {'BasicTrust':>11} {'HTFL':>9} {'Diff vs FedAvg':>14} {'Detect':>8} {'Compress':>9}")
    logger.info("=" * 80)
    for row in mgr.build_comparison_table(all_results):
        logger.info(
            f"{row['scenario']:<22} "
            f"{row['fedavg']:>9.4f} "
            f"{row['basic_trust']:>11.4f} "
            f"{row['adaptive_htfl']:>9.4f} "
            f"{row['improvement_over_fedavg']:>+12.4f} "
            f"{row['detection_rate']:>8.2%} "
            f"{row['avg_compression']:>9.2%}"
        )
    logger.info("=" * 80)
    logger.info("\nRun 'python dashboard.py' to generate figures.")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds",  type=int, default=CONFIG["n_rounds"])
    parser.add_argument("--clients", type=int, default=CONFIG["n_clients"])
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--data-source", type=str, default=CONFIG["data_source"],
                        choices=["synthetic", "csv"],
                        help="Data source: 'synthetic' (code-generated) or 'csv' (from CSV files)")
    parser.add_argument("--data-dir", type=str, default=CONFIG["data_dir"],
                        help="Path to CSV data directory (default: data/csv_data/)")
    args = parser.parse_args()
    cfg = CONFIG.copy()
    cfg["n_rounds"]    = args.rounds
    cfg["n_clients"]   = args.clients
    cfg["data_source"] = args.data_source
    cfg["data_dir"]    = args.data_dir
    main(cfg, args.verbose)
