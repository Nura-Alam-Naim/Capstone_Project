"""
Delegated Proof-of-Trust (DPoT) Micro-chain — Adaptive-HTFL
============================================================
Pillar 3: Lightweight smart-contract-style consensus architecture.

Architecture:
  - Edge servers with aggregated client trust ABOVE a dynamic threshold
    are elected to the consensus COMMITTEE.
  - Only committee members may vote to finalise the global model.
  - Each round produces a lightweight Block containing:
      * round number
      * committee members (elected edge servers)
      * model fingerprint (hash of aggregated weights)
      * trust snapshot
      * timestamp (simulated)
  - Non-committee servers observe but cannot vote.
  - Dynamic threshold adapts based on network health.

This eliminates the latency of traditional global blockchain consensus
by restricting finality to a small, trusted committee rather than
requiring all n nodes to agree.
"""

import numpy as np
import hashlib
import json
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict


@dataclass
class DPoTBlock:
    """A single block in the DPoT micro-chain."""
    round_num: int
    committee: List[int]             # elected edge server IDs
    excluded: List[int]              # below-threshold servers
    model_fingerprint: str           # SHA-256 of aggregated weights
    trust_snapshot: Dict[int, float] # trust scores at this round
    dynamic_threshold: float         # threshold used for committee election
    consensus_votes: int             # number of committee votes cast
    consensus_reached: bool          # True if >50% committee voted yes
    timestamp: float = field(default_factory=time.time)
    block_hash: str = ""

    def compute_hash(self) -> str:
        content = json.dumps({
            "round": self.round_num,
            "committee": sorted(self.committee),
            "fingerprint": self.model_fingerprint,
            "threshold": round(self.dynamic_threshold, 4),
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def finalise(self):
        self.block_hash = self.compute_hash()


@dataclass
class SmartContract:
    """
    Simulated smart contract governing committee election rules.
    In a real deployment this would run on each edge server.
    """
    base_threshold: float = 0.55       # minimum trust to join committee
    min_committee_size: int = 3        # always elect at least this many
    consensus_quorum: float = 0.51     # fraction of committee needed to finalize

    def compute_dynamic_threshold(
        self, trust_scores: Dict[int, float], round_num: int
    ) -> float:
        """
        Dynamic threshold: starts at base, tightens as network matures.
        Also adapts if the network is under heavy attack (mean trust drops).
        """
        mean_trust = float(np.mean(list(trust_scores.values())))
        # Tighten threshold as rounds progress (network learns to self-police)
        maturity_factor = min(0.10, round_num * 0.003)
        # Relax threshold if mean trust is low (attack scenario, avoid empty committee)
        attack_factor = max(0, (0.5 - mean_trust) * 0.3)
        threshold = self.base_threshold + maturity_factor - attack_factor
        return float(np.clip(threshold, 0.30, 0.80))

    def elect_committee(
        self,
        trust_scores: Dict[int, float],
        round_num: int,
    ) -> Tuple[List[int], List[int], float]:
        """
        Elect committee from servers above dynamic threshold.
        Always ensures minimum committee size by taking top-N if needed.
        """
        threshold = self.compute_dynamic_threshold(trust_scores, round_num)
        all_ids = sorted(trust_scores.keys())

        above = [cid for cid in all_ids if trust_scores[cid] >= threshold]
        below = [cid for cid in all_ids if trust_scores[cid] < threshold]

        # Guarantee minimum committee size
        if len(above) < self.min_committee_size:
            ranked = sorted(all_ids, key=lambda c: trust_scores[c], reverse=True)
            above = ranked[:self.min_committee_size]
            below = [c for c in all_ids if c not in above]

        return above, below, threshold

    def simulate_vote(
        self, committee: List[int], trust_scores: Dict[int, float]
    ) -> Tuple[int, bool]:
        """
        Simulate committee vote. Higher-trust members vote yes with higher probability.
        Returns (votes_cast, consensus_reached).
        """
        votes = 0
        for cid in committee:
            # Probability of yes vote proportional to trust score
            p_yes = float(np.clip(trust_scores.get(cid, 0.5), 0, 1))
            if np.random.random() < p_yes:
                votes += 1
        quorum_needed = max(1, int(len(committee) * self.consensus_quorum))
        return votes, votes >= quorum_needed


class DPoTChain:
    """
    Delegated Proof-of-Trust Micro-chain.
    Maintains the append-only chain of consensus blocks.
    """

    def __init__(self, contract: Optional[SmartContract] = None):
        self.contract = contract or SmartContract()
        self.chain: List[DPoTBlock] = []
        self.pending_model: Optional[Dict] = None

    def fingerprint_weights(self, weights: Dict[str, np.ndarray]) -> str:
        """Compute a short fingerprint of the aggregated model weights."""
        flat = np.concatenate([v.flatten() for v in weights.values()])
        h = hashlib.sha256(flat.tobytes()).hexdigest()[:16]
        return h

    def propose_round(
        self,
        round_num: int,
        global_weights: Dict[str, np.ndarray],
        trust_scores: Dict[int, float],
    ) -> DPoTBlock:
        """
        Run one DPoT consensus round:
          1. Elect committee based on trust scores
          2. Simulate committee vote
          3. Create and append block
        """
        np.random.seed(round_num)  # deterministic for reproducibility

        # Smart contract: elect committee
        committee, excluded, threshold = self.contract.elect_committee(
            trust_scores, round_num
        )

        # Committee votes on the proposed model
        votes, consensus = self.contract.simulate_vote(committee, trust_scores)

        # Create block
        block = DPoTBlock(
            round_num=round_num,
            committee=committee,
            excluded=excluded,
            model_fingerprint=self.fingerprint_weights(global_weights),
            trust_snapshot={k: round(v, 4) for k, v in trust_scores.items()},
            dynamic_threshold=round(threshold, 4),
            consensus_votes=votes,
            consensus_reached=consensus,
        )
        block.finalise()
        self.chain.append(block)
        return block

    def get_chain_summary(self) -> List[Dict]:
        return [
            {
                "round": b.round_num,
                "committee_size": len(b.committee),
                "excluded": len(b.excluded),
                "threshold": b.dynamic_threshold,
                "votes": b.consensus_votes,
                "consensus": b.consensus_reached,
                "fingerprint": b.model_fingerprint,
                "hash": b.block_hash,
            }
            for b in self.chain
        ]

    def verify_chain_integrity(self) -> bool:
        """Verify that all block hashes are correct (tamper detection)."""
        for block in self.chain:
            if block.block_hash != block.compute_hash():
                return False
        return True

    def latest_block(self) -> Optional[DPoTBlock]:
        return self.chain[-1] if self.chain else None

    def consensus_rate(self) -> float:
        if not self.chain:
            return 0.0
        return sum(1 for b in self.chain if b.consensus_reached) / len(self.chain)

    def average_committee_size(self) -> float:
        if not self.chain:
            return 0.0
        return float(np.mean([len(b.committee) for b in self.chain]))
