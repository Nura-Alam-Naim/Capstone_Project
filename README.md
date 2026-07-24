# Adaptive-HTFL
### Hybrid Trust-Aware Federated Learning Framework
#### Secure Smart Campus IoT Networks

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Overview

**Adaptive-HTFL** is a novel federated learning framework designed for secure, resource-constrained smart campus IoT deployments. It addresses three critical limitations of existing FL systems:

| Limitation | Our Solution |
|---|---|
| High computational overhead of trust mechanisms | Lightweight PCA-based latent-space trust |
| Susceptibility of fixed trust ratings | Temporal Trust Decay with adaptive scoring |
| Cannot distinguish malicious vs non-IID data | Latent-Space cosine similarity separation |

---

## Three Core Architectural Pillars

### 1. Multi-Dimensional Dynamic Trust Scoring
- **Contextual/Hardware Trust** — monitors battery health, network stability, CPU load
- **Latent-Space Data Trust** — PCA dimensionality reduction + cosine similarity to distinguish poisoning from natural non-IID variation
- **Temporal Trust Decay** — decaying function emphasises recent behaviour, prevents historical reliability masking late-stage attacks

### 2. Resource-Aware Adaptive Sparsification
- Gradient compression level is dynamically linked to each node's real-time trust score
- High-trust nodes send compressed updates (bandwidth-efficient)
- Low-trust/unverified nodes are forced to send full updates (for closer inspection)

### 3. Delegated Proof-of-Trust (DPoT) Micro-chain
- Lightweight smart-contract-style consensus architecture
- Only edge servers with aggregated trust above a dynamic threshold join the consensus committee
- Eliminates latency of traditional global blockchain consensus

---

## Project Structure

```
adaptive_htfl/
├── core/
│   ├── client.py              # IoT client node with hardware context simulation
│   ├── server.py              # Adaptive-HTFL aggregation server
│   └── trust_engine_ai.py        # Multi-Dimensional Dynamic Trust Scoring
├── blockchain/
│   └── dpot_chain.py          # Delegated Proof-of-Trust micro-chain
├── attacks/
│   └── attack_simulator.py    # Label flip, noise, scaling, slow poison
├── data/
│   └── iot_data_generator.py  # Synthetic smart campus IoT sensor data
├── evaluation/
│   └── metrics.py             # Accuracy, detection rate, sparsification stats
├── utils/
│   └── logger.py              # Structured logging
├── results/                   # JSON results + figures
├── run_experiment.py          # Main experiment runner
├── dashboard.py               # Publication-quality figures
└── requirements.txt
```

## Quick Start

```bash
pip install -r requirements.txt
python run_experiment.py
python dashboard.py
```

## Compared Strategies

| Strategy | Description |
|---|---|
| **FedAvg** | Standard baseline — no trust, no compression |
| **Basic Trust-FL** | Simple cosine similarity trust (prior work) |
| **Adaptive-HTFL** | Our full framework — all three pillars active |

## Dataset

Synthetic smart campus IoT sensor data:
- 10 sensor modalities: temperature, humidity, CO2, motion, light, air quality, sound, occupancy, power, vibration
- 10 activity classes across campus zones
- Non-IID distribution across clients (Dirichlet α=0.5)
- Hardware context metadata per client (battery, signal, CPU)

## References

- McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data" (FedAvg)
- Blanchard et al., "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent"
- Fung et al., "Mitigating Sybils in Federated Learning Poisoning"
- Nakamoto, S., "Bitcoin: A Peer-to-Peer Electronic Cash System" (blockchain inspiration)
