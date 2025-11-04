"""Lightweight LATS (Local Agent Tree Search) prototype.

This module provides a small, dependency-free tree search component that
agents can use to build and persist reasoning trees. It is intentionally
minimal: a few helpers for UCT-style selection, expansion, simulation
(placeholder) and backpropagation. The goal is to provide a pluggable
API agents can call during inference; later you can replace "simulate"
with actual agent rollouts or learned value estimators.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class LATSNode:
    id: str
    state: Dict[str, Any]
    parent: Optional['LATSNode'] = None
    children: List['LATSNode'] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    last_updated: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'state': self.state,
            'visits': self.visits,
            'value': self.value,
            'children': [c.to_dict() for c in self.children],
        }

    @staticmethod
    def from_dict(d: Dict[str, Any], parent: Optional['LATSNode'] = None) -> 'LATSNode':
        node = LATSNode(id=d['id'], state=d.get('state', {}), parent=parent)
        node.visits = d.get('visits', 0)
        node.value = d.get('value', 0.0)
        for c in d.get('children', []):
            child = LATSNode.from_dict(c, parent=node)
            node.children.append(child)
        return node


class LATS:
    """Simple tree search manager.

    Usage pattern (example):
        tree = LATS(root_state)
        for _ in range(n_simulations):
            node = tree.select()
            child = tree.expand(node, new_state)
            reward = tree.simulate(child)  # placeholder
            tree.backpropagate(child, reward)

    This design keeps simulate() as a hook that can be overridden or
    swapped out with agent-specific rollouts.
    """

    def __init__(self, root_state: Dict[str, Any], root_id: str = "root"):
        self.root = LATSNode(id=root_id, state=root_state)
        self.nodes: Dict[str, LATSNode] = {self.root.id: self.root}

    def uct_score(self, parent: LATSNode, child: LATSNode, c: float = 1.414) -> float:
        if child.visits == 0:
            return float('inf')
        exploit = child.value / child.visits
        explore = c * math.sqrt(math.log(max(1, parent.visits)) / child.visits)
        return exploit + explore

    def select(self) -> LATSNode:
        """Select node to expand using UCT from root."""
        node = self.root
        while node.children:
            # choose child with max UCT
            scores = [self.uct_score(node, c) for c in node.children]
            max_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
            node = node.children[max_idx]
        return node

    def expand(self, parent: LATSNode, child_state: Dict[str, Any], child_id: Optional[str] = None) -> LATSNode:
        """Add a child node to parent with given state."""
        if child_id is None:
            child_id = f"n{len(self.nodes)}"
        node = LATSNode(id=child_id, state=child_state, parent=parent)
        parent.children.append(node)
        self.nodes[child_id] = node
        return node

    def simulate(self, node: LATSNode) -> float:
        """Placeholder simulation. Replace with agent rollouts or learned model.

        Returns a scalar reward/value.
        """
        # Simple heuristic: random value biased by a small heuristic in state
        bias = 0.0
        if isinstance(node.state, dict):
            bias = float(node.state.get('heuristic', 0.0))
        # Simulate a noisy outcome
        return float(max(0.0, min(1.0, bias + random.gauss(0.5, 0.15))))

    def backpropagate(self, node: LATSNode, reward: float) -> None:
        cur = node
        while cur is not None:
            cur.visits += 1
            cur.value += reward
            cur.last_updated = time.time()
            cur = cur.parent

    def best_child(self, node: Optional[LATSNode] = None) -> Optional[LATSNode]:
        if node is None:
            node = self.root
        if not node.children:
            return None
        # choose child with highest average value
        best = max(node.children, key=lambda c: (c.value / max(1, c.visits)))
        return best

    def choose(self, exploit: float = 0.9) -> LATSNode:
        """High-level choose action: with probability exploit pick best_child, else run a sim and expand."""
        if random.random() < exploit:
            best = self.best_child()
            if best:
                return best
        # exploration path: perform a selection, expansion, simulation and backprop
        node = self.select()
        # create a trivial new child state; in practice this should be produced by agent
        new_state = {'heuristic': random.random()}
        child = self.expand(node, new_state)
        reward = self.simulate(child)
        self.backpropagate(child, reward)
        return child

    def to_json(self) -> str:
        return json.dumps(self.root.to_dict(), indent=2)

    def save(self, path: str) -> None:
        with open(path, 'w') as f:
            f.write(self.to_json())

    @staticmethod
    def load(path: str) -> 'LATS':
        with open(path, 'r') as f:
            d = json.load(f)
        root = LATSNode.from_dict(d)
        tree = LATS(root_state=root.state, root_id=root.id)
        # rebuild nodes mapping
        def walk(n: LATSNode):
            tree.nodes[n.id] = n
            for c in n.children:
                c.parent = n
                walk(c)
        walk(root)
        tree.root = root
        return tree
