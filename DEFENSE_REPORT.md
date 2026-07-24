# Adaptive-HTFL Defense Report

Project title: Adaptive-HTFL: A Hybrid Trust-Aware Federated Learning Framework for Secure Smart Campus IoT Networks

Document purpose: This report explains the complete concept, technical design, workflow, implementation, methods, experiment logic, strengths, and limitations of the current Adaptive-HTFL prototype so that a reader can understand the project end to end.

Prepared for: Project defense / viva / presentation

## 1. Executive Summary

Adaptive-HTFL is a federated learning prototype designed for smart campus IoT environments where many distributed devices collaborate to train a global model without sending raw data to a central server. The main problem it addresses is that standard federated learning can fail in realistic campus settings because devices are resource-constrained, data are naturally non-IID, and some clients may send malicious or poisoned updates.

The project proposes a hybrid solution with three main ideas:

1. Multi-Dimensional Dynamic Trust Scoring
2. Resource-Aware Adaptive Sparsification
3. Delegated Proof-of-Trust (DPoT) Micro-chain

The current implementation is a simulation-based research prototype. It shows how trust, compression, and lightweight consensus can be combined in one framework. It is suitable as a proof-of-concept for academic defense, but it should be presented honestly as a prototype rather than a full production deployment.

## 2. Problem Statement

Traditional federated learning assumes that client updates are mostly honest and that the aggregation server can average them safely. In smart campus IoT networks, this assumption is weak because:

1. Devices have unstable hardware conditions such as low battery, weak signal, or high CPU load.
2. The local data distribution across campus locations is naturally non-IID.
3. Some clients may launch poisoning attacks such as label flipping, noise injection, scaling attacks, or slow poison attacks.
4. Full model updates from all devices can consume too much bandwidth.
5. Blockchain-style security can become too heavy for low-latency IoT environments.

The core research question is:

How can a federated learning system become more secure, adaptive, and bandwidth-efficient in a smart campus IoT network while still distinguishing malicious behavior from natural non-IID variation?

## 3. Research Gap from Literature

The project is built around three limitations commonly discussed in the literature:

1. High computational overhead of traditional trust mechanisms
2. Fixed trust ratings that do not adapt fast enough to recent behavior
3. Difficulty separating malicious poisoning from benign non-IID behavior

Adaptive-HTFL answers this gap by combining lightweight trust computation, temporal trust updating, latent-space analysis, trust-aware communication control, and a lightweight consensus simulation.

## 4. Proposed Solution

Adaptive-HTFL is a hybrid framework where the server does not treat every client equally. Instead, every client update is evaluated using multiple signals:

1. Hardware/context trust
2. Latent-space behavioral trust
3. Temporal reputation

These trust scores influence two decisions:

1. How much a client contributes to the global aggregation
2. How strongly a client is compressed before transmission

After aggregation, a DPoT micro-chain simulates a lightweight trust-driven consensus process for validating the global model.

## 5. High-Level Architecture

The architecture has six main layers:

1. Synthetic smart campus data layer
2. Client training layer
3. Attack simulation layer
4. Trust scoring and weighted aggregation layer
5. DPoT consensus layer
6. Evaluation and visualization layer

End-to-end flow:

```text
Synthetic IoT Dataset
        ->
Non-IID Client Partitions
        ->
Local Client Training
        ->
Optional Attack Injection
        ->
Trust-Aware Sparsification
        ->
Server Trust Scoring
        ->
Trust-Weighted Aggregation
        ->
DPoT Committee Validation
        ->
Global Model Evaluation
        ->
Saved Results and Dashboard Figures
```

## 6. Full Workflow of the Project

### Step 1: Dataset generation

The project first creates a synthetic smart campus IoT dataset. Ten sensor modalities are simulated:

1. temperature
2. humidity
3. CO2
4. motion
5. light
6. air quality
7. sound
8. occupancy
9. power
10. vibration

The dataset models ten campus activity classes such as empty room, lecture class, lab session, cafeteria activity, corridor traffic, and emergency condition.

This is implemented in `data/iot_data_generator.py`.

### Step 2: Data normalization and non-IID partitioning

The full dataset is normalized and then split across clients using a Dirichlet distribution with `alpha = 0.5`. This simulates non-IID behavior, meaning each client sees a different local distribution of classes.

This is important because real campus devices do not all observe the same environment. For example, a classroom sensor and a cafeteria sensor naturally produce different data.

### Step 3: Hardware context generation

Each client is assigned hardware metadata such as:

1. battery percentage
2. signal strength
3. CPU load
4. uptime
5. packet loss

Honest clients get healthy profiles, while malicious or faulty clients can receive suspicious or degraded profiles. This metadata is later used by the hardware trust score.

### Step 4: Local client initialization

Each client is created with:

1. local training data
2. a simple local model
3. hardware context
4. optional attack type
5. attack parameters
6. current trust score from the server

The client uses a local softmax classifier and performs several local epochs of SGD-style training.

### Step 5: Local training and attack injection

A client receives the current global model, trains locally, and computes its update. If the client is malicious, the project can inject one of four attacks:

1. Label flip attack
2. Noise injection attack
3. Scaling attack
4. Slow poisoning attack

This simulates adversarial behavior before the update is sent to the server.

### Step 6: Resource-aware adaptive sparsification

After local training, the client compresses its update. The compression level depends on the trust score sent by the server in the previous round.

The implemented logic is:

1. High trust leads to high sparsification
2. Low trust leads to little or no sparsification

This means trusted nodes send fewer parameters and save bandwidth, while suspicious nodes are forced to reveal more of their update for server inspection.

### Step 7: Server-side trust scoring

The server collects all client updates and computes a trust score for each client using three components:

1. Hardware trust
2. Latent-space trust
3. Temporal reputation

These components are combined into a final trust score.

### Step 8: Trust-weighted aggregation

The server uses trust scores and local sample counts to compute aggregation weights. A client with more data and higher trust contributes more to the global model than a client with low trust.

### Step 9: DPoT consensus

After aggregation, the framework simulates a delegated consensus layer. A dynamic threshold selects trusted members for a committee. The committee votes on the aggregated global model. A block is created containing committee membership, excluded nodes, model fingerprint, threshold, and consensus status.

### Step 10: Evaluation and dashboard generation

The project evaluates final accuracy, convergence behavior, attack detection statistics, compression behavior, and DPoT consensus statistics. Results are stored in JSON and can be visualized with a dashboard and plots.

## 7. Project Modules and What Each Part Does

### `run_experiment.py`

This is the main execution script. It:

1. Loads configuration
2. Generates the dataset
3. Creates non-IID client partitions
4. Builds attack scenarios
5. Runs the three strategies
6. Logs round-by-round results
7. Saves the experiment output

The three compared strategies are:

1. `FedAvg`
2. `BasicTrust`
3. `AdaptiveHTFL`

### `core/client.py`

This file defines:

1. `LocalModel`
2. `IoTClient`

`LocalModel` is a simple softmax classifier implemented manually with:

1. linear projection
2. softmax activation
3. cross-entropy loss
4. gradient-based parameter update

`IoTClient` is responsible for:

1. receiving the global model
2. local training
3. attack injection
4. adaptive sparsification
5. sending metadata back to the server

### `core/server.py`

This file defines the aggregation server. It:

1. receives client weights and metadata
2. computes flat updates
3. invokes the trust engine
4. computes trust-weighted aggregation coefficients
5. updates the global model
6. triggers DPoT consensus
7. stores trust and compression history

### `core/trust_engine_ai.py`

This is the central intelligence module of the project. It implements the multi-dimensional trust engine.

It supports:

1. hardware trust scoring
2. PCA-based latent trust
3. autoencoder-based latent trust
4. IsolationForest anomaly scoring
5. optional logistic regression for supervised trust prediction
6. temporal trust decay and reputation update
7. trust-based aggregation weights

This is the main place where AI is used in the project.

### `blockchain/dpot_chain.py`

This file implements the Delegated Proof-of-Trust simulation. It defines:

1. `DPoTBlock`
2. `SmartContract`
3. `DPoTChain`

The file handles:

1. dynamic threshold calculation
2. committee election
3. vote simulation
4. model fingerprint hashing
5. block creation
6. chain integrity checking

### `data/iot_data_generator.py`

This file generates:

1. synthetic smart campus sensor data
2. hardware metadata
3. non-IID client partitions

This allows the project to simulate a realistic campus environment without requiring a real deployment dataset.

### `attacks/attack_simulator.py`

This file defines the available attack scenarios and their parameters. It decides:

1. which clients are malicious
2. what attack they perform
3. attack severity settings

### `evaluation/metrics.py`

This file calculates:

1. final accuracy
2. convergence round
3. detection rate
4. false positive rate
5. average compression
6. DPoT consensus rate

It also saves experiment results in JSON format.

### `dashboard.py`

This file produces visual outputs such as:

1. convergence plot
2. final accuracy bar chart
3. trust heatmap
4. robustness radar chart
5. DPoT and compression figure

### `run.bat`

This is the Windows automation script. It:

1. runs the experiment
2. generates dashboard figures
3. starts a local web server for viewing the dashboard

### `watch_dashboard.py`

This is an optional utility script that watches code changes and reruns the pipeline automatically. It is a development convenience feature, not part of the main algorithm.

## 8. Technical Methods Used in the Project

### 8.1 Federated learning base model

The local learning model is a softmax classifier. For each client:

1. input features are multiplied by weight matrix `W`
2. bias `b` is added
3. softmax converts scores into class probabilities
4. cross-entropy loss is computed
5. weights are updated by local training

This keeps the model simple, lightweight, and appropriate for a prototype.

### 8.2 Non-IID data simulation

The project uses Dirichlet partitioning to make client datasets heterogeneous. This is a standard way to simulate realistic federated environments.

Why it matters:

1. It tests whether the trust mechanism can avoid confusing non-IID diversity with malicious behavior.
2. It reflects the fact that different campus areas have different sensor patterns.

### 8.3 Hardware trust

Hardware trust is derived from five signals:

1. battery
2. signal strength
3. CPU load
4. uptime
5. packet loss

These are converted to bounded scores and combined as a weighted sum.

Interpretation:

1. Low battery may indicate unstable edge operation.
2. High CPU load may indicate overload or suspicious activity.
3. Poor network quality may explain noisy or delayed updates.
4. Very short uptime can indicate frequent restart behavior.
5. High packet loss suggests unreliable communication.

This helps the server avoid over-trusting updates from physically unstable nodes.

### 8.4 Latent-space trust

The trust engine supports two main styles:

1. Baseline latent trust using PCA and cosine similarity
2. Enhanced latent trust using an autoencoder and anomaly detection

PCA path:

1. client updates are projected into a lower-dimensional space
2. the center of the latent space is computed
3. cosine similarity and distance from the center are used as trust indicators

Autoencoder path:

1. a small one-hidden-layer autoencoder is trained on a reference bank of updates
2. each update is encoded into latent form
3. reconstruction error is measured
4. latent distance from the reference center is measured
5. IsolationForest optionally detects anomalies in latent space

Optional supervised path:

1. if trust labels are available, logistic regression can learn a benign-vs-malicious separation
2. this path is present in code but not actively used in the default experiment

Why this is called AI:

1. the autoencoder is a learned neural model
2. IsolationForest is an ML anomaly detector
3. logistic regression is a supervised ML classifier

### 8.5 Temporal trust decay

Trust is not static. The system keeps a reputation score for every client and updates it each round using temporal decay.

Purpose:

1. recent behavior matters more than old behavior
2. previously honest nodes can still be downgraded if they become suspicious
3. repeated anomalies trigger stronger penalties

This addresses the weakness of fixed trust mechanisms.

### 8.6 Final trust formulation

The implemented trust formula is:

```text
T_i = alpha * HardwareTrust_i
    + beta  * LatentTrust_i
    + gamma * TemporalReputation_i
```

Where the default coefficients are:

1. `alpha = 0.30`
2. `beta = 0.45`
3. `gamma = 0.25`

This means latent-space behavior has the strongest influence, followed by hardware context and temporal reputation.

### 8.7 Trust-weighted aggregation

After computing trust, the server calculates aggregation weights:

```text
AggregationWeight_i = (Trust_i * Samples_i) / Sum(Trust_j * Samples_j)
```

This gives more influence to clients that are both trustworthy and data-rich.

### 8.8 Adaptive sparsification

The client computes a compression ratio from trust:

```text
compression_ratio = trust_score * 0.90
```

Then top-k sparsification is applied:

1. only the largest-magnitude update components are kept
2. the rest are zeroed out

Interpretation:

1. trust score `1.0` means up to `90%` sparsification
2. trust score `0.0` means full dense update

This reduces bandwidth for trusted nodes and gives more visibility into suspicious nodes.

### 8.9 DPoT micro-chain

The DPoT mechanism is a lightweight consensus simulation inspired by blockchain principles but simplified for edge environments.

Main logic:

1. compute a dynamic committee threshold
2. elect nodes above the threshold
3. guarantee a minimum committee size
4. simulate committee votes
5. create a block with the model fingerprint and trust snapshot

The threshold adapts based on:

1. round maturity
2. mean trust of the network

This is meant to avoid the cost of a full blockchain consensus across all participants.

## 9. Attack Scenarios Used

The experiment includes five scenarios:

1. Baseline without attack
2. Label flip attack
3. Noise injection attack
4. Scaling / Byzantine attack
5. Slow poisoning attack

Why these attacks matter:

1. Label flip tests data poisoning
2. Noise injection tests random corruption
3. Scaling tests extreme gradient manipulation
4. Slow poisoning tests stealthy long-term attack behavior

This gives the framework a varied evaluation environment.

## 10. Evaluation Metrics

The project tracks several metrics:

1. Final accuracy
2. Accuracy history over rounds
3. Convergence round
4. Detection rate
5. False positive rate
6. Average compression
7. DPoT consensus rate

What these metrics mean:

1. Final accuracy measures model quality.
2. Convergence round measures learning speed.
3. Detection rate measures how many malicious nodes were correctly flagged.
4. False positive rate measures how many honest nodes were wrongly flagged.
5. Average compression measures bandwidth saving.
6. DPoT consensus rate measures how often the committee accepted a round.

## 11. What Makes This Project "Adaptive"

The word adaptive is important in the project title. It is justified because:

1. trust changes over time instead of staying fixed
2. compression changes based on trust
3. the DPoT threshold changes based on network condition and maturity
4. the reference pool of trusted updates changes over rounds

So the project is not using a static defense. It changes its behavior as the system evolves.

## 12. What Part Uses AI

The AI portion is centered in the trust engine.

AI methods used:

1. Autoencoder for learned latent representation and reconstruction scoring
2. IsolationForest for anomaly detection in latent space
3. LogisticRegression for optional supervised benign-vs-malicious scoring

Why AI is placed here:

1. trust evaluation is the hardest decision problem in the framework
2. the trust engine must separate benign non-IID variation from malicious updates
3. learned latent analysis is more defensible than pure fixed heuristics

Other modules remain mostly rule-based because:

1. that keeps the prototype computationally light
2. it makes the architecture easier to explain
3. it isolates AI to the most meaningful component

## 13. Implemented vs Simulated vs Not Yet Fully Realized

This section is important for defense honesty.

### Fully implemented in the prototype

1. Synthetic dataset generation
2. Non-IID partitioning
3. Client local training
4. Attack injection
5. Multi-dimensional trust score computation
6. Trust-weighted aggregation
7. Trust-aware adaptive sparsification
8. DPoT committee election and block generation
9. Result logging and plotting

### Simulated rather than fully real-world deployed

1. Campus data are synthetic, not collected from real university devices
2. Hardware context is simulated
3. DPoT is a software simulation, not a distributed blockchain deployment
4. Clients and server run in one experimental environment, not on physically distributed campus hardware

### Present in code but not fully exploited by the default experiment

1. Supervised latent trust scoring is available but not enabled by default
2. Trust labels are supported by the server API but are not actively passed in the current client metadata

### Important implementation limitations in the current prototype

1. `FedAvg` and `BasicTrust` are not perfectly isolated baselines because client-side sparsification still exists in the shared client logic.
2. DPoT currently runs after aggregation and logs consensus, but it does not yet block or roll back a model update when consensus fails.
3. Detection depends on a fixed flagging threshold, and in existing runs malicious clients are often downweighted rather than hard-flagged.
4. Two dashboard figures are illustrative:
   Figure 3 creates a synthetic trust heatmap for presentation.
   Figure 5 uses approximate compression values instead of plotting all real stored compression dynamics.
5. The current logger can show Unicode issues in Windows terminal output because some log messages use special characters.

These points do not invalidate the prototype, but they should be admitted clearly during defense.

## 14. Strengths of the Project

1. The architecture directly follows a clear research motivation.
2. The project integrates trust, compression, and consensus in one framework.
3. It uses AI in the most relevant part of the system.
4. It addresses realistic smart campus concerns such as non-IID data and resource constraints.
5. It provides a reproducible simulation environment.
6. It includes comparison against baseline strategies.
7. It produces interpretable outputs and figures for academic presentation.

## 15. Main Limitations

1. The local model is intentionally simple and not state-of-the-art deep learning.
2. The dataset is synthetic.
3. DPoT is simulated, not deployed across real edge servers.
4. Malicious detection is stronger as soft weighting than as strict binary blocking.
5. The current code is more of a research prototype than a production system.

## 16. How to Defend the Prototype Correctly

The best defense position is:

Adaptive-HTFL is a practical research prototype that demonstrates the feasibility of combining AI-based trust analysis, trust-aware communication control, and lightweight delegated consensus in a smart campus federated learning setting.

Do not oversell it as:

1. a real campus deployment
2. a production blockchain system
3. a fully validated attack detector with field data

Do present it as:

1. an end-to-end prototype
2. an implementation-backed framework
3. a simulation study with architectural novelty
4. a foundation for future real deployment work

## 17. Defense-Friendly Explanation of the Full Workflow

If asked to explain the system in one complete flow, say:

Adaptive-HTFL first generates smart campus sensor data and splits them across IoT clients in a non-IID way. Each client receives the current global model, trains locally, and may behave honestly or maliciously depending on the selected scenario. Before sending its update, the client compresses the update according to its trust score. The server then evaluates each client using hardware signals, latent-space AI analysis, and temporal reputation. These trust values determine how much each client contributes to the global model. After aggregation, a DPoT committee is elected from high-trust participants to validate the model and record a lightweight trust block. Finally, the system evaluates accuracy, robustness, detection behavior, and communication efficiency, and produces plots for analysis.

## 18. Commands to Run the Project

Main commands:

```bat
pip install -r requirements.txt
python run_experiment.py
python dashboard.py
```

Windows batch automation:

```bat
run.bat
```

This batch file:

1. runs the experiment
2. builds the dashboard
3. serves the dashboard on `http://localhost:8000/index.html`

## 19. Suggested Presentation Structure

A strong presentation order is:

1. Problem and motivation
2. Research gap
3. Proposed Adaptive-HTFL framework
4. Three architectural pillars
5. Full system workflow
6. Technical methods
7. Experiment setup and attacks
8. Results and interpretation
9. Strengths and limitations
10. Conclusion and future work

## 20. Short Defense Summary

Adaptive-HTFL is a federated learning prototype for secure smart campus IoT networks. Its key innovation is that it does not trust all clients equally. Instead, it combines AI-based latent-space trust, hardware-aware trust, temporal trust decay, adaptive communication compression, and lightweight delegated consensus. The implementation demonstrates that this hybrid design can be studied end to end in a reproducible experimental environment. The current system is a strong proof-of-concept prototype, with clear paths for future real-world deployment improvements.

## 21. Future Work

The most logical next steps are:

1. Use a real smart campus dataset or real device telemetry
2. Replace the simple local model with a stronger deep model
3. Make DPoT enforce model acceptance rather than just log consensus
4. Separate baselines more cleanly so `FedAvg`, `BasicTrust`, and `AdaptiveHTFL` differ exactly as intended
5. Improve binary malicious detection beyond fixed thresholding
6. Learn hardware trust and compression policy directly from data
7. Deploy the framework across real edge devices or edge containers

## 22. Viva Questions You Should Be Ready For

1. Why did you choose federated learning instead of centralized learning?
2. Why is non-IID data a challenge in smart campus IoT?
3. What makes the trust engine multi-dimensional?
4. Why is an autoencoder appropriate here?
5. How does your system separate poisoning from benign heterogeneity?
6. Why is trust connected to compression?
7. How is DPoT different from a traditional blockchain?
8. What part of your framework is actually AI?
9. What are the biggest limitations of the prototype?
10. If you had six more months, what would you improve first?

## 23. Best One-Line Defense Statement

Adaptive-HTFL is an implementation-backed smart campus federated learning prototype that combines AI-driven trust analysis, trust-aware communication efficiency, and lightweight delegated consensus to improve security and robustness under non-IID and adversarial conditions.
