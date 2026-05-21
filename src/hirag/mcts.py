from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

import networkx as nx


def _edge_key(source: str, target: str) -> tuple[str, str]:
    return tuple(sorted((source, target)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _estimate_text_tokens(value: str) -> int:
    text = (value or "").strip()
    if not text:
        return 0
    return max(1, len(text.split()))


@dataclass(frozen=True)
class MCTSBridgeConfig:
    max_iterations: int = 96
    exploration_constant: float = 1.2
    candidate_hops: int = 2
    candidate_top_neighbors: int = 12
    progressive_widening_coefficient: float = 2.0
    progressive_widening_exponent: float = 0.5
    rollout_top_k: int = 3
    rollout_depth: int = 4
    rollout_epsilon: float = 0.1
    max_path_edges: int = 120
    max_token_budget: int = 12500
    length_penalty: float = 0.02
    token_penalty: float = 0.15
    target_reward: float = 1.0


@dataclass
class _PathState:
    current: str
    path: list[str]
    visited_nodes: set[str]
    token_cost: int

    @property
    def depth(self) -> int:
        return max(0, len(self.path) - 1)


@dataclass
class _TreeNode:
    state: _PathState
    target: str
    parent: "_TreeNode | None" = None
    action_from_parent: str | None = None
    visit_count: int = 0
    total_reward: float = 0.0
    children: dict[str, "_TreeNode"] = field(default_factory=dict)
    ordered_actions: list[str] = field(default_factory=list)
    terminal_reason: str | None = None

    def average_reward(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_reward / self.visit_count


def _edge_cost(
    edge_costs: dict[tuple[str, str], dict[str, float]], source: str, target: str
) -> float:
    return edge_costs.get(_edge_key(source, target), {}).get("cost", 1.0)


def _edge_utility(
    edge_costs: dict[tuple[str, str], dict[str, float]], source: str, target: str
) -> float:
    return max(0.0, 1.0 - _edge_cost(edge_costs, source, target))


def _edge_token_cost(graph: nx.Graph, source: str, target: str) -> int:
    edge_data = graph.get_edge_data(source, target, default={}) or {}
    return _estimate_text_tokens(str(edge_data.get("description", "")))


def _build_candidate_subgraph(
    graph: nx.Graph,
    source: str,
    target: str,
    edge_costs: dict[tuple[str, str], dict[str, float]],
    config: MCTSBridgeConfig,
) -> nx.Graph:
    seed_nodes = {source, target}
    candidate_nodes = {source, target}
    for seed in seed_nodes:
        if not graph.has_node(seed):
            continue
        lengths = nx.single_source_shortest_path_length(graph, seed, cutoff=config.candidate_hops)
        candidate_nodes.update(lengths.keys())

    candidate_graph = graph.subgraph(candidate_nodes).copy()
    pruned_graph = nx.Graph()
    pruned_graph.add_nodes_from(candidate_graph.nodes(data=True))

    for node in candidate_graph.nodes():
        neighbors = list(candidate_graph.neighbors(node))
        neighbors.sort(
            key=lambda neighbor: (
                _edge_cost(edge_costs, node, neighbor),
                candidate_graph.degree(neighbor),
                neighbor,
            )
        )
        keep = set(neighbors[: max(1, config.candidate_top_neighbors)])
        if source in neighbors:
            keep.add(source)
        if target in neighbors:
            keep.add(target)
        for neighbor in keep:
            pruned_graph.add_edge(node, neighbor, **(candidate_graph.get_edge_data(node, neighbor) or {}))

    if pruned_graph.has_node(source) and pruned_graph.has_node(target):
        try:
            nx.shortest_path(pruned_graph, source=source, target=target)
            return pruned_graph
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass
    return candidate_graph


def _neighbor_order(
    graph: nx.Graph,
    current: str,
    target: str,
    visited_nodes: set[str],
    edge_costs: dict[tuple[str, str], dict[str, float]],
) -> list[str]:
    try:
        distance_map = nx.single_source_shortest_path_length(graph, target)
    except Exception:
        distance_map = {}

    def score(neighbor: str) -> tuple[float, float, float, str]:
        revisit_penalty = 1.0 if neighbor in visited_nodes else 0.0
        target_distance = distance_map.get(neighbor)
        proximity_bonus = 0.0 if target_distance is None else 1.0 / (1.0 + target_distance)
        return (
            -_edge_utility(edge_costs, current, neighbor) - 0.4 * proximity_bonus + 0.2 * revisit_penalty,
            graph.degree(neighbor),
            _edge_cost(edge_costs, current, neighbor),
            neighbor,
        )

    return sorted(graph.neighbors(current), key=score)


def _allowed_expansions(node: _TreeNode, config: MCTSBridgeConfig) -> int:
    visits = max(1, node.visit_count)
    limit = config.progressive_widening_coefficient * (visits ** config.progressive_widening_exponent)
    return max(1, int(limit))


def _is_terminal(node: _TreeNode, config: MCTSBridgeConfig) -> str | None:
    if node.state.current == node.target:
        return "target_reached"
    if node.state.depth >= config.max_path_edges:
        return "depth_limit"
    if node.state.token_cost >= config.max_token_budget:
        return "token_budget"
    if not node.ordered_actions:
        return "dead_end"
    return None


def _expand(node: _TreeNode, graph: nx.Graph, edge_costs: dict[tuple[str, str], dict[str, float]], config: MCTSBridgeConfig) -> _TreeNode:
    if not node.ordered_actions:
        return node
    expansion_limit = _allowed_expansions(node, config)
    unexpanded = [action for action in node.ordered_actions if action not in node.children]
    if not unexpanded or len(node.children) >= expansion_limit:
        return node
    action = unexpanded[0]
    token_cost = node.state.token_cost + _edge_token_cost(graph, node.state.current, action)
    child_state = _PathState(
        current=action,
        path=node.state.path + [action],
        visited_nodes=node.state.visited_nodes | {action},
        token_cost=token_cost,
    )
    child = _TreeNode(
        state=child_state,
        target=node.target,
        parent=node,
        action_from_parent=action,
    )
    child.ordered_actions = _neighbor_order(
        graph, action, node.target, child_state.visited_nodes, edge_costs
    )
    child.terminal_reason = _is_terminal(child, config)
    node.children[action] = child
    return child


def _select_child(node: _TreeNode, config: MCTSBridgeConfig) -> _TreeNode:
    log_parent_visits = math.log(max(1, node.visit_count))
    best_child: _TreeNode | None = None
    best_score = float("-inf")
    for child in node.children.values():
        if child.visit_count == 0:
            score = float("inf")
        else:
            exploration = config.exploration_constant * math.sqrt(
                log_parent_visits / child.visit_count
            )
            score = child.average_reward() + exploration
        if score > best_score:
            best_score = score
            best_child = child
    return best_child or node


def _rollout(
    node: _TreeNode,
    graph: nx.Graph,
    edge_costs: dict[tuple[str, str], dict[str, float]],
    config: MCTSBridgeConfig,
    rng: random.Random,
) -> tuple[list[str], str]:
    current = node.state.current
    path = list(node.state.path)
    visited = set(node.state.visited_nodes)
    token_cost = node.state.token_cost
    rollout_steps = 0

    while rollout_steps < config.rollout_depth:
        if current == node.target:
            return path, "target_reached"
        if len(path) - 1 >= config.max_path_edges:
            return path, "depth_limit"
        if token_cost >= config.max_token_budget:
            return path, "token_budget"
        candidates = [
            neighbor
            for neighbor in _neighbor_order(graph, current, node.target, visited, edge_costs)
            if neighbor not in visited
        ]
        if not candidates:
            return path, "dead_end"
        top_candidates = candidates[: max(1, config.rollout_top_k)]
        if len(top_candidates) == 1 or rng.random() > config.rollout_epsilon:
            next_node = top_candidates[0]
        else:
            next_node = rng.choice(top_candidates)
        token_cost += _edge_token_cost(graph, current, next_node)
        path.append(next_node)
        visited.add(next_node)
        current = next_node
        rollout_steps += 1

    if current == node.target:
        return path, "target_reached"
    return path, "rollout_limit"


def _path_reward(
    graph: nx.Graph,
    path: list[str],
    target: str,
    edge_costs: dict[tuple[str, str], dict[str, float]],
    config: MCTSBridgeConfig,
) -> float:
    if len(path) < 2:
        return config.target_reward if path and path[-1] == target else 0.0
    edge_utilities = [
        _edge_utility(edge_costs, left, right)
        for left, right in zip(path, path[1:])
    ]
    mean_edge_utility = sum(edge_utilities) / len(edge_utilities) if edge_utilities else 0.0
    token_cost = sum(_edge_token_cost(graph, left, right) for left, right in zip(path, path[1:]))
    token_ratio = (
        min(1.0, token_cost / config.max_token_budget)
        if config.max_token_budget > 0
        else 0.0
    )
    reward = mean_edge_utility
    if path[-1] == target:
        reward += config.target_reward
    reward -= config.length_penalty * max(0, len(path) - 1)
    reward -= config.token_penalty * token_ratio
    return float(reward)


def find_mcts_bridge_path(
    graph: nx.Graph,
    source: str,
    target: str,
    edge_costs: dict[tuple[str, str], dict[str, float]],
    config: MCTSBridgeConfig,
) -> tuple[list[str], dict[str, Any]]:
    if not graph.has_node(source) or not graph.has_node(target):
        return [], {"decision": "mcts_no_path", "reason": "missing_node"}

    candidate_graph = _build_candidate_subgraph(graph, source, target, edge_costs, config)
    if not candidate_graph.has_node(source) or not candidate_graph.has_node(target):
        return [], {"decision": "mcts_no_path", "reason": "candidate_subgraph_missing_node"}
    try:
        nx.shortest_path(candidate_graph, source=source, target=target)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return [], {
            "decision": "mcts_no_path",
            "reason": "candidate_subgraph_disconnected",
            "candidate_nodes": candidate_graph.number_of_nodes(),
            "candidate_edges": candidate_graph.number_of_edges(),
        }

    rng = random.Random(f"{source}|{target}|{candidate_graph.number_of_nodes()}")
    root = _TreeNode(
        state=_PathState(current=source, path=[source], visited_nodes={source}, token_cost=0),
        target=target,
    )
    root.ordered_actions = _neighbor_order(candidate_graph, source, target, {source}, edge_costs)
    root.terminal_reason = _is_terminal(root, config)

    best_path: list[str] = []
    best_reward = float("-inf")
    fallback_path = nx.dijkstra_path(
        candidate_graph,
        source=source,
        target=target,
        weight=lambda u, v, _data: _edge_cost(edge_costs, u, v),
    )
    fallback_reward = _path_reward(candidate_graph, fallback_path, target, edge_costs, config)
    iteration_count = 0
    successful_rollouts = 0

    for _ in range(config.max_iterations):
        iteration_count += 1
        node = root
        while True:
            if node.terminal_reason is not None:
                break
            if len(node.children) < min(len(node.ordered_actions), _allowed_expansions(node, config)):
                node = _expand(node, candidate_graph, edge_costs, config)
                break
            if not node.children:
                node.terminal_reason = _is_terminal(node, config)
                break
            node = _select_child(node, config)

        rollout_path, rollout_reason = _rollout(node, candidate_graph, edge_costs, config, rng)
        reward = _path_reward(candidate_graph, rollout_path, target, edge_costs, config)
        if rollout_path and rollout_path[-1] == target:
            successful_rollouts += 1
        if reward > best_reward:
            best_reward = reward
            best_path = rollout_path

        cursor: _TreeNode | None = node
        while cursor is not None:
            cursor.visit_count += 1
            cursor.total_reward += reward
            cursor = cursor.parent

        if rollout_reason == "target_reached" and reward >= fallback_reward:
            break

    if best_path and best_path[-1] == target and best_reward >= fallback_reward:
        return best_path, {
            "decision": "mcts",
            "iterations": iteration_count,
            "successful_rollouts": successful_rollouts,
            "best_reward": best_reward,
            "fallback_reward": fallback_reward,
            "candidate_nodes": candidate_graph.number_of_nodes(),
            "candidate_edges": candidate_graph.number_of_edges(),
        }

    return fallback_path, {
        "decision": "mcts_weighted_fallback",
        "iterations": iteration_count,
        "successful_rollouts": successful_rollouts,
        "best_reward": best_reward if best_reward != float("-inf") else None,
        "fallback_reward": fallback_reward,
        "candidate_nodes": candidate_graph.number_of_nodes(),
        "candidate_edges": candidate_graph.number_of_edges(),
    }
