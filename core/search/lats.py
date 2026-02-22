from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

@dataclass
class LATSNode:
    id: str
    state: Dict[str, Any]
    parent: Optional['LATSNode'] = None
    children: List['LATSNode'] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    last_updated: float = field(default_factory=time.time)
    pruned: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Iterative serialization to avoid stack overflow."""
        result = {
            'id': self.id,
            'state': self.state,
            'visits': self.visits,
            'value': self.value,
            'pruned': self.pruned,
            'children': []
        }
        
        stack = [(self, result['children'])]
        
        while stack:
            current_node, parent_children_list = stack.pop()
            
            for child in current_node.children:
                child_dict = {
                    'id': child.id,
                    'state': child.state,
                    'visits': child.visits,
                    'value': child.value,
                    'pruned': child.pruned,
                    'children': []
                }
                parent_children_list.append(child_dict)
                
                if child.children:
                    stack.append((child, child_dict['children']))
        
        return result

    @staticmethod
    def from_dict(d: Dict[str, Any], parent: Optional['LATSNode'] = None) -> 'LATSNode':
        """Iterative deserialization to avoid stack overflow."""
        root = LATSNode(
            id=d['id'],
            state=d.get('state', {}),
            parent=parent,
            visits=d.get('visits', 0),
            value=d.get('value', 0.0),
            pruned=d.get('pruned', False)
        )
        
        queue = [(root, d)]
        
        while queue:
            current_node, current_dict = queue.pop(0)
            
            for child_dict in current_dict.get('children', []):
                child = LATSNode(
                    id=child_dict['id'],
                    state=child_dict.get('state', {}),
                    parent=current_node,
                    visits=child_dict.get('visits', 0),
                    value=child_dict.get('value', 0.0),
                    pruned=child_dict.get('pruned', False)
                )
                current_node.children.append(child)
                queue.append((child, child_dict))
        
        return root

class LATS:
    def __init__(self, root_state: Dict[str, Any], root_id: str = "root", *,
                 simulator: Optional[callable] = None,
                 expand_fn: Optional[callable] = None,
                 coherence_checker: Optional[callable] = None,
                 exploration_constant: float = 1.414,
                 prune_visit_threshold: int = 3,
                 prune_reward_threshold: float = 0.3):
        self.root = LATSNode(id=root_id, state=root_state)
        self.nodes: Dict[str, LATSNode] = {self.root.id: self.root}
        self.simulator = simulator
        self.expand_fn = expand_fn
        self.coherence_checker = coherence_checker
        self.exploration_constant = exploration_constant
        self.prune_visit_threshold = prune_visit_threshold
        self.prune_reward_threshold = prune_reward_threshold

    def uct_score(self, parent: LATSNode, child: LATSNode, c: float = 1.414) -> float:
        c = self.exploration_constant if c is None else c
        if child.visits == 0:
            return float('inf')
        exploit = child.value / child.visits
        explore = c * math.sqrt(math.log(max(1, parent.visits)) / child.visits)
        return exploit + explore

    def select(self) -> LATSNode:
        import logging
        logger = logging.getLogger("kaelum.lats")
        
        node = self.root
        depth = 0
        while node.children:
            unpruned = [c for c in node.children if not c.pruned]
            if not unpruned:
                logger.debug(f"LATS-SELECT: All children pruned at depth {depth}")
                break
            
            scores = [self.uct_score(node, c) for c in unpruned]
            max_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
            selected_child = unpruned[max_idx]
            
            avg_reward = selected_child.value / max(1, selected_child.visits)
            logger.debug(f"LATS-SELECT: Depth {depth} → Node {selected_child.id} "
                        f"(UCT={scores[max_idx]:.3f}, visits={selected_child.visits}, "
                        f"avg_reward={avg_reward:.3f}, {len(unpruned)} unpruned children)")
            
            node = selected_child
            depth += 1
        
        logger.debug(f"LATS-SELECT: Leaf node {node.id} selected at depth {depth}")
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
            raise NotImplementedError("LATS.simulate requires a simulator callable.")
        
        is_terminal = len(node.children) == 0 and node.state.get("answer")
        
        if not is_terminal and self.coherence_checker:
            coherent = self.coherence_checker(node)
            if not coherent:
                import logging
                logger = logging.getLogger("kaelum.lats")
                logger.debug(f"LATS-VERIFY: Node {node.id} failed coherence check, applying penalty")
                return -0.5
        
        reward = sim(node)
        try:
            return float(reward)
        except Exception as e:
            raise RuntimeError(f"Simulator must return a numeric reward: {e}")

    def backpropagate(self, node: LATSNode, reward: float) -> None:
        import logging
        logger = logging.getLogger("kaelum.lats")
        
        cur = node
        depth = 0
        while cur is not None:
            cur.visits += 1
            cur.value += reward
            cur.last_updated = time.time()
            
            avg_reward = cur.value / cur.visits
            
            # Early pruning: Stop exploring branches that show poor performance
            # Thresholds are configurable; higher visit threshold yields more confidence before pruning.
            if (self.prune_visit_threshold is not None and
                cur.visits >= self.prune_visit_threshold and
                self.prune_reward_threshold is not None and
                avg_reward < self.prune_reward_threshold):
                if not cur.pruned:
                    cur.pruned = True
                    logger.info(f"LATS-PRUNE: Node {cur.id} pruned at depth {depth} "
                               f"(visits={cur.visits}, avg_reward={avg_reward:.3f} < 0.3)")
            
            cur = cur.parent
            depth += 1

    def best_child(self, node: Optional['LATSNode'] = None) -> Optional['LATSNode']:
        if node is None:
            node = self.root
        if not node.children:
            return None
        return max(node.children, key=lambda c: (c.value / max(1, c.visits)))
    
    def get_avg_reward(self) -> float:
        if not self.nodes:
            return 0.0
        total_visits = sum(n.visits for n in self.nodes.values())
        if total_visits == 0:
            return 0.0
        total_value = sum(n.value for n in self.nodes.values())
        return total_value / total_visits

    def run_simulations(self, num_simulations: int, max_depth: int = 10, parallel: bool = False, max_workers: int = 4):
        import logging
        logger = logging.getLogger("kaelum.lats")
        
        logger.info(f"LATS: Starting {num_simulations} simulations (max_depth={max_depth}, parallel={parallel})")
        start_time = time.time()
        
        if parallel and num_simulations >= 4:
            logger.info(f"LATS: Using parallel execution with {max_workers} workers")
            self._run_parallel_simulations(num_simulations, max_depth, max_workers)
        else:
            logger.info(f"LATS: Using sequential execution")
            for i in range(num_simulations):
                if i % 5 == 0 and i > 0:
                    logger.debug(f"LATS: Completed {i}/{num_simulations} simulations")
                self._run_single_simulation(max_depth)
        
        elapsed = time.time() - start_time
        avg_reward = self.get_avg_reward()
        total_nodes = len(self.nodes)
        logger.info(f"LATS: ✓ Simulations complete in {elapsed:.2f}s")
        logger.info(f"LATS: Total nodes explored: {total_nodes}")
        logger.info(f"LATS: Average reward: {avg_reward:.3f}")
    
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
        import logging
        logger = logging.getLogger("kaelum.lats")
        failed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._run_single_simulation, max_depth) for _ in range(num_simulations)]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    failed += 1
                    logger.warning(f"LATS: Simulation failed ({failed} total failures): {e}")
        if failed:
            logger.warning(f"LATS: {failed}/{num_simulations} parallel simulations failed")

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
