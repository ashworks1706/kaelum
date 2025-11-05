from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed


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
    def __init__(self, root_state: Dict[str, Any], root_id: str = "root", *,
                 simulator: Optional[callable] = None,
                 expand_fn: Optional[callable] = None):
        self.root = LATSNode(id=root_id, state=root_state)
        self.nodes: Dict[str, LATSNode] = {self.root.id: self.root}
        self.simulator = simulator
        self.expand_fn = expand_fn

    def uct_score(self, parent: LATSNode, child: LATSNode, c: float = 1.414) -> float:
        if child.visits == 0:
            return float('inf')
        exploit = child.value / child.visits
        explore = c * math.sqrt(math.log(max(1, parent.visits)) / child.visits)
        return exploit + explore

    def select(self) -> LATSNode:
        node = self.root
        while node.children:
            scores = [self.uct_score(node, c) for c in node.children]
            max_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
            node = node.children[max_idx]
        return node

    def expand(self, parent: LATSNode, child_state: Dict[str, Any], child_id: Optional[str] = None) -> LATSNode:
        if child_id is None:
            child_id = f"n{len(self.nodes)}"
        node = LATSNode(id=child_id, state=child_state, parent=parent)
        parent.children.append(node)
        self.nodes[child_id] = node
        return node

    def simulate(self, node: LATSNode, simulator: Optional[callable] = None) -> float:
        sim = simulator or self.simulator
        if sim is None:
            raise NotImplementedError(
                "LATS.simulate requires a simulator callable. "
                "Provide it as LATS(simulator=...) or pass it to simulate(node, simulator=...)."
            )
        reward = sim(node)
        try:
            return float(reward)
        except Exception as e:
            raise RuntimeError(f"Simulator must return a numeric reward: {e}")

    def backpropagate(self, node: LATSNode, reward: float) -> None:
        cur = node
        while cur is not None:
            cur.visits += 1
            cur.value += reward
            cur.last_updated = time.time()
            cur = cur.parent

    def best_child(self, node: Optional['LATSNode'] = None) -> Optional['LATSNode']:
        if node is None:
            node = self.root
        if not node.children:
            return None
        best = max(node.children, key=lambda c: (c.value / max(1, c.visits)))
        return best

    def run_simulations(self, num_simulations: int, max_depth: int = 10, parallel: bool = False, max_workers: int = 4):
        if parallel and num_simulations >= 4:
            self._run_parallel_simulations(num_simulations, max_depth, max_workers)
        else:
            for _ in range(num_simulations):
                self._run_single_simulation(max_depth)
    
    def _run_single_simulation(self, max_depth: int):
        node = self.select()
        if node.state.get("depth", 0) >= max_depth:
            return
        if self.expand_fn:
            child_state = self.expand_fn(node)
            child = self.expand(node, child_state)
        else:
            child = node
        reward = self.simulate(child)
        self.backpropagate(child, reward)
    
    def _run_parallel_simulations(self, num_simulations: int, max_depth: int, max_workers: int):
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._run_single_simulation, max_depth) for _ in range(num_simulations)]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    pass

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
        def walk(n: LATSNode):
            tree.nodes[n.id] = n
            for c in n.children:
                c.parent = n
                walk(c)
        walk(root)
        tree.root = root
        return tree
