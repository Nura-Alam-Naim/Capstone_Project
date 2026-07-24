"""
IoT Client Node — Adaptive-HTFL
Each client simulates a smart campus IoT device cluster.

New features vs basic Trust-FL:
  - Hardware context metadata (battery, signal, CPU load, uptime, packet loss)
  - Resource-Aware Adaptive Sparsification: gradient compression inversely
    linked to trust score. High-trust nodes send sparse updates; low-trust
    nodes send full dense updates for closer inspection by the server.
"""

import numpy as np
from typing import Tuple, Optional, Dict
from sklearn.preprocessing import LabelBinarizer


class LocalModel:
    """
    Softmax classifier: W [n_features × n_classes], b [n_classes].
    Hand-coded forward pass, cross-entropy loss, SGD.
    """

    def __init__(self, n_features: int, n_classes: int, lr: float = 0.05):
        self.n_features = n_features
        self.n_classes = n_classes
        self.lr = lr
        self.W = np.zeros((n_features, n_classes))
        self.b = np.zeros(n_classes)
        self.lb = LabelBinarizer()
        self.lb.fit(list(range(n_classes)))

    def softmax(self, z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def forward(self, X):
        return self.softmax(X @ self.W + self.b)

    def train_step(self, X, y):
        n = len(X)
        probs = self.forward(X)
        Y = self.lb.transform(y)
        loss = -np.mean(np.sum(Y * np.log(probs + 1e-8), axis=1))
        dZ = probs - Y
        self.W -= self.lr * (X.T @ dZ / n)
        self.b -= self.lr * dZ.mean(axis=0)
        return loss

    def train_epochs(self, X, y, epochs=5, batch_size=32):
        n, loss = len(X), 0.0
        for _ in range(epochs):
            idx = np.random.permutation(n)
            for s in range(0, n, batch_size):
                b = idx[s:s + batch_size]
                loss = self.train_step(X[b], y[b])
        return loss

    def predict(self, X):
        return self.forward(X).argmax(axis=1)

    def accuracy(self, X, y):
        return float(np.mean(self.predict(X) == y))

    def get_weights(self):
        return {"W": self.W.copy(), "b": self.b.copy()}

    def set_weights(self, weights):
        self.W = weights["W"].copy()
        self.b = weights["b"].copy()

    def get_flat_weights(self):
        return np.concatenate([self.W.flatten(), self.b.flatten()])

    def set_flat_weights(self, flat):
        ws = self.n_features * self.n_classes
        self.W = flat[:ws].reshape(self.n_features, self.n_classes)
        self.b = flat[ws:]


class IoTClient:
    """
    Smart campus IoT FL client with:
      - Hardware context metadata
      - Adaptive gradient sparsification based on trust score
      - Data/model attack simulation
    """

    def __init__(
        self,
        client_id: int,
        X_train: np.ndarray,
        y_train: np.ndarray,
        hardware_context: Dict[str, float],
        n_features: int = 10,
        n_classes: int = 10,
        local_epochs: int = 5,
        learning_rate: float = 0.05,
        attack_type: Optional[str] = None,
        attack_params: Optional[Dict] = None,
    ):
        self.client_id = client_id
        self.X_train = X_train
        self.y_train = y_train
        self.hardware_context = hardware_context
        self.n_features = n_features
        self.n_classes = n_classes
        self.local_epochs = local_epochs
        self.attack_type = attack_type
        self.attack_params = attack_params or {}
        self.model = LocalModel(n_features, n_classes, learning_rate)
        self.is_malicious = attack_type is not None
        self.n_samples = len(X_train)

        # Track updates + compression stats
        self.update_history = []
        self.compression_history = []   # sparsification ratios used
        self.current_trust_score = 1.0  # updated by server each round

    def receive_global_model(self, global_weights: Dict):
        self.model.set_weights(global_weights)

    def set_trust_score(self, score: float):
        """Server informs client of its current trust score for sparsification."""
        self.current_trust_score = float(np.clip(score, 0.0, 1.0))

    def local_train(self, enable_compression: bool = True) -> Tuple[Dict, int, float, Dict]:
        """
        Train locally, apply attacks if malicious, apply adaptive sparsification.

        Returns:
            weights       : model weight dict (potentially sparsified)
            n_samples     : number of local training samples
            loss          : final training loss
            meta          : metadata dict (compression_ratio, hardware_context)
        """
        pre_flat = self.model.get_flat_weights().copy()

        # Apply data-level attacks
        X_tr, y_tr = self._apply_data_attack(self.X_train, self.y_train)

        loss = self.model.train_epochs(X_tr, y_tr, epochs=self.local_epochs)

        post_flat = self.model.get_flat_weights()
        update = post_flat - pre_flat

        # Apply model-level attacks
        update = self._apply_model_attack(update)

        # ── Pillar 2: Resource-Aware Adaptive Sparsification ──
        # High-trust nodes: high compression (send less)
        # Low-trust nodes:  low compression (send full update for inspection)
        if enable_compression:
            compression_ratio = self._compute_compression_ratio(self.current_trust_score)
        else:
            compression_ratio = 0.0
            
        update = self._sparsify(update, compression_ratio)

        self.update_history.append(update.copy())
        self.compression_history.append(compression_ratio)

        # Reconstruct weights from sparsified update
        sparsified_flat = pre_flat + update
        self.model.set_flat_weights(sparsified_flat)

        meta = {
            "compression_ratio": compression_ratio,
            "hardware_context": self.hardware_context,
            "is_malicious": self.is_malicious,
            "attack_type": self.attack_type,
            "trust_label": int(self.is_malicious),
        }
        return self.model.get_weights(), self.n_samples, loss, meta

    def _compute_compression_ratio(self, trust_score: float) -> float:
        """
        Adaptive sparsification ratio based on trust (linear).
        trust=1.0 → 10% sparsification (send top 90% gradients)
        trust=0.5 → 5% sparsification (send top 95% gradients)
        trust=0.0 → 0% sparsification (send full update for inspection)
        """
        # Linear mapping: very light compression for small models
        max_sparsity = 0.10
        return float(trust_score * max_sparsity)

    def _sparsify(self, update: np.ndarray, ratio: float) -> np.ndarray:
        """
        Top-k sparsification: keep only the (1-ratio) fraction of
        largest-magnitude gradients. Others zeroed out.
        """
        if ratio <= 0.0:
            return update
        n = len(update)
        k = max(1, int(n * (1.0 - ratio)))
        top_k_idx = np.argpartition(np.abs(update), -k)[-k:]
        sparse = np.zeros_like(update)
        sparse[top_k_idx] = update[top_k_idx]
        return sparse

    def _apply_data_attack(self, X, y):
        if self.attack_type == "label_flip":
            y = y.copy()
            src = self.attack_params.get("target_class", 0)
            dst = self.attack_params.get("flip_to", 1)
            y[y == src] = dst
        elif self.attack_type == "slow_poison":
            y = y.copy()
            rate = self.attack_params.get("poison_rate", 0.15)
            n_p = int(len(y) * rate)
            idx = np.random.choice(len(y), n_p, replace=False)
            y[idx] = np.random.randint(0, self.n_classes, n_p)
        return X, y

    def _apply_model_attack(self, update):
        if self.attack_type == "noise":
            scale = self.attack_params.get("noise_scale", 3.0)
            std = np.std(update) if np.std(update) > 0 else 1.0
            update += np.random.normal(0, scale * std, update.shape)
        elif self.attack_type == "scaling":
            update *= self.attack_params.get("scale_factor", 10.0)
        return update

    def evaluate(self, X_test, y_test):
        return self.model.accuracy(X_test, y_test)

    def get_recent_update(self):
        return self.update_history[-1] if self.update_history else None

    def __repr__(self):
        tag = f"[{self.attack_type.upper()}]" if self.is_malicious else "[HONEST]"
        return f"IoTClient(id={self.client_id}, n={self.n_samples}, trust={self.current_trust_score:.2f}, {tag})"
