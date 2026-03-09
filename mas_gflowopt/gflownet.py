from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_utils import GraphOp, apply_op, is_acyclic, legal_ops, make_empty_dag, op_to_str
from .representation import GraphRepresentationModel
from .reward import MASTaskEvaluator, MASRewardModel
from .scoring import ScoreModel
from .types import (
    DAGState,
    GFlowNetTrainingStats,
    MASConfig,
    RewardBreakdown,
    SampledDAG,
    TrajectoryStep,
)

Vector = List[float]


def _log_prob(p: float) -> float:
    return math.log(max(p, 1e-12))


@dataclass
class _StateContext:
    dag: DAGState
    z: Vector
    question_vector: Vector
    src_vecs: Dict[int, Vector]
    dst_vecs: Dict[int, Vector]
    edge_ops: List[GraphOp]
    edge_cond_probs: List[float]  # P(edge | not-stop)
    probs: List[float]  # edge actions + stop action
    stop_index: int
    reward_breakdown: RewardBreakdown

    @property
    def stop_prob(self) -> float:
        return self.probs[self.stop_index]

    @property
    def reward(self) -> float:
        return self.reward_breakdown.reward

    @property
    def log_reward(self) -> float:
        return _log_prob(self.reward)


@dataclass
class _TransitionRecord:
    step_id: int
    cur: _StateContext
    nxt: _StateContext
    action_index: int
    op: GraphOp
    log_pf_action: float
    log_pb: float

class _PolicyNet(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(dim * 8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.stop_mlp = nn.Sequential(
            nn.Linear(dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.ctx_edge = nn.Linear(dim, 1, bias=False)
        self.ctx_stop = nn.Linear(dim, 1, bias=False)
        self.ctx_q_edge = nn.Linear(dim, 1, bias=False)
        self.ctx_q_stop = nn.Linear(dim, 1, bias=False)

    def edge_scores(self, z: torch.Tensor, q: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        if src.ndim == 1:
            src = src.unsqueeze(0)
        if dst.ndim == 1:
            dst = dst.unsqueeze(0)
        z_e = z.unsqueeze(0).expand(src.size(0), -1)
        q_e = q.unsqueeze(0).expand(src.size(0), -1)
        feat = torch.cat(
            [
                src,
                dst,
                z_e,
                q_e,
                src * dst,
                q_e * src,
                q_e * dst,
                q_e * z_e,
            ],
            dim=-1,
        )
        return self.edge_mlp(feat).squeeze(-1)

    def stop_logit(self, z: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        qz = z * q
        feat = torch.cat([z, q, qz], dim=-1)
        return self.stop_mlp(feat).squeeze(-1)

    def context_embed(self, z: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        qz = z * q
        e = self.ctx_edge(z.unsqueeze(0)).squeeze(0)
        s = self.ctx_stop(z.unsqueeze(0)).squeeze(0)
        qe = self.ctx_q_edge(qz.unsqueeze(0)).squeeze(0)
        qs = self.ctx_q_stop(qz.unsqueeze(0)).squeeze(0)
        return torch.stack([e, s, qe, qs], dim=0).squeeze(-1)


class GFlowNetSampler:
    """Trainable GFlowNet-style DAG sampler.

    Implemented objectives:
    - Detailed-balance loss for forward transitions.
    - Contrastive loss head (NT-Xent style with dot-product logits).
    """

    def __init__(
        self,
        config: MASConfig,
        repr_model: GraphRepresentationModel,
        scorer: ScoreModel,
        reward_model: Optional[MASRewardModel] = None,
    ):
        self.config = config
        self.repr_model = repr_model
        self.scorer = scorer
        self.reward_model = reward_model or MASRewardModel(config, scorer)

        self.rng = random.Random(config.random_seed)
        dim = config.embedding_dim
        hidden_dim = max(8, config.policy_hidden_dim)
        self.policy = _PolicyNet(dim, hidden_dim)
        proj_dim = max(2, config.cl_proj_dim)
        self.proj = nn.Linear(dim, proj_dim, bias=False)
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self._initialized = False

    def initialize(self) -> None:
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(
                list(self.policy.parameters()) + list(self.proj.parameters()),
                lr=self.config.gflownet_lr,
                weight_decay=self.config.gflownet_weight_decay,
            )
        self._initialized = True

    def _forward_edge_ops(self, dag: DAGState) -> List[GraphOp]:
        mode = self.config.gflownet_action_space.lower()
        if mode == "edge_add":
            # Keep DB assumption aligned with edge-add construction.
            ops = legal_ops(dag, allow_backtracking=False, allow_reverse=False)
            return [op for op in ops if op[0] == "add"]
        return legal_ops(
            dag,
            allow_backtracking=self.config.allow_backtracking,
            allow_reverse=self.config.allow_backtracking,
        )

    def _parent_state_count(self, next_dag: DAGState) -> int:
        """Number of parent states under the configured forward action grammar."""
        mode = self.config.gflownet_action_space.lower()
        signatures: Set[Tuple[Tuple[int, int], ...]] = set()
        edge_set = set(next_dag.edges)
        n = len(next_dag.nodes)

        # Parent via forward-add: remove one existing edge.
        for e in edge_set:
            prev_edges = sorted(edge_set - {e})
            signatures.add(tuple(prev_edges))

        if mode == "extended":
            # Parent via forward-delete: predecessor has one extra edge.
            if self.config.allow_backtracking:
                for i in range(n):
                    for j in range(n):
                        if i == j or (i, j) in edge_set:
                            continue
                        cand = sorted(edge_set | {(i, j)})
                        if is_acyclic(next_dag.nodes, cand):
                            signatures.add(tuple(cand))

            # Parent via forward-reverse.
            if self.config.allow_backtracking:
                for src, dst in list(edge_set):
                    rev = (dst, src)
                    if rev in edge_set:
                        continue
                    cand = set(edge_set)
                    cand.discard((src, dst))
                    cand.add(rev)
                    cand_sorted = sorted(cand)
                    if is_acyclic(next_dag.nodes, cand_sorted):
                        signatures.add(tuple(cand_sorted))

        return max(1, len(signatures))

    def _should_true_eval(self, step_id: int, eval_calls: int, force_terminal: bool) -> bool:
        if force_terminal and self.config.true_eval_terminal_always:
            return True
        interval = max(0, self.config.true_eval_interval)
        budget = max(0, self.config.true_eval_budget_per_trajectory)
        if interval == 0:
            return False
        if eval_calls >= budget:
            return False
        return step_id % interval == 0

    def _tensor(self, x: Sequence[float]) -> torch.Tensor:
        return torch.tensor(list(x), dtype=torch.float32)

    def _policy_tensors(
        self,
        z: Vector,
        q: Vector,
        src_vecs: Dict[int, Vector],
        dst_vecs: Dict[int, Vector],
        edge_ops: List[GraphOp],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_t = self._tensor(z)
        q_t = self._tensor(q)
        if not edge_ops:
            return torch.empty(0), z_t, q_t
        src = self._tensor([src_vecs[src] for _, src, _ in edge_ops])
        dst = self._tensor([dst_vecs[dst] for _, _, dst in edge_ops])
        scores = self.policy.edge_scores(z_t, q_t, src, dst)
        return scores, z_t, q_t

    def _policy_distribution(
        self,
        z: Vector,
        q: Vector,
        src_vecs: Dict[int, Vector],
        dst_vecs: Dict[int, Vector],
        edge_ops: List[GraphOp],
    ) -> Tuple[List[float], List[float], float]:
        if not edge_ops:
            return [1.0], [], 1.0
        with torch.no_grad():
            edge_scores, z_t, q_t = self._policy_tensors(z, q, src_vecs, dst_vecs, edge_ops)
            edge_cond_probs_t = F.softmax(edge_scores, dim=0)
            stop_prob_t = torch.sigmoid(self.policy.stop_logit(z_t, q_t))
            edge_cond_probs = edge_cond_probs_t.cpu().tolist()
            stop_prob = float(stop_prob_t.item())
        probs = [(1.0 - stop_prob) * p for p in edge_cond_probs]
        probs.append(stop_prob)
        return probs, edge_cond_probs, stop_prob

    def _build_state_context(
        self,
        dag: DAGState,
        agent_vectors: Dict[str, Vector],
        prev_z: Optional[Vector],
        evaluator: Optional[MASTaskEvaluator],
        use_true_eval: bool,
        question_text: Optional[str],
        question_vector: Optional[Vector],
    ) -> _StateContext:
        z, src_vecs, dst_vecs = self.repr_model.encode_graph(dag, agent_vectors, prev_z=prev_z)
        dag_with_z = DAGState(nodes=list(dag.nodes), edges=list(dag.edges), z=z, reward=dag.reward)
        if question_vector is None:
            q = [0.0] * len(z)
        elif len(question_vector) == len(z):
            q = question_vector
        elif len(question_vector) > len(z):
            q = question_vector[: len(z)]
        else:
            q = question_vector + [0.0] * (len(z) - len(question_vector))

        edge_ops = self._forward_edge_ops(dag_with_z)
        probs, edge_cond_probs, _ = self._policy_distribution(z, q, src_vecs, dst_vecs, edge_ops)

        rb = self.reward_model.score_and_reward(
            dag_with_z,
            evaluator=evaluator,
            use_true_evaluator=use_true_eval,
            question_text=question_text,
            question_vector=q,
            agent_vectors=agent_vectors,
        )
        dag_with_z.reward = rb.reward

        return _StateContext(
            dag=dag_with_z,
            z=z,
            question_vector=q,
            src_vecs=src_vecs,
            dst_vecs=dst_vecs,
            edge_ops=edge_ops,
            edge_cond_probs=edge_cond_probs,
            probs=probs,
            stop_index=len(edge_ops),
            reward_breakdown=rb,
        )

    def _weighted_choice_idx(self, probs: List[float]) -> int:
        if not probs:
            return 0
        total = sum(probs)
        if total <= 0:
            return self.rng.randrange(0, len(probs))
        r = self.rng.random() * total
        c = 0.0
        for i, p in enumerate(probs):
            c += p
            if c >= r:
                return i
        return len(probs) - 1

    def _sample_one_with_records(
        self,
        nodes: List[str],
        agent_vectors: Dict[str, Vector],
        evaluator: Optional[MASTaskEvaluator],
        question_text: Optional[str],
        question_vector: Optional[Vector],
        task_tag: Optional[str] = None,
    ) -> Tuple[SampledDAG, List[_TransitionRecord]]:
        if not self._initialized:
            raise RuntimeError("GFlowNetSampler.initialize() must be called first.")

        dag = make_empty_dag(nodes)
        transitions: List[_TransitionRecord] = []
        trajectory: List[TrajectoryStep] = []
        prev_z: Optional[Vector] = None
        terminal_ctx: Optional[_StateContext] = None
        eval_calls = 0

        for step in range(self.config.gflownet_max_steps):
            use_true_cur = evaluator is not None and self._should_true_eval(
                step_id=step,
                eval_calls=eval_calls,
                force_terminal=False,
            )
            cur = self._build_state_context(
                dag,
                agent_vectors,
                prev_z=prev_z,
                evaluator=evaluator,
                use_true_eval=use_true_cur,
                question_text=question_text,
                question_vector=question_vector,
            )
            if use_true_cur:
                eval_calls += 1
            terminal_ctx = cur

            if not cur.edge_ops:
                trajectory.append(
                    TrajectoryStep(
                        step_id=step,
                        action="stop(no-legal-op)",
                        graph=cur.dag,
                        reward=0.0 if self.config.step_reward_signal == "delta_log_reward" else (cur.reward if self.config.reward_every_step else 0.0),
                        z=cur.z,
                        op=None,
                        log_pf=_log_prob(cur.stop_prob),
                        log_pb=0.0,
                        stop_prob=cur.stop_prob,
                    )
                )
                break

            action_index = self._weighted_choice_idx(cur.probs)
            if action_index == cur.stop_index:
                trajectory.append(
                    TrajectoryStep(
                        step_id=step,
                        action="stop(policy)",
                        graph=cur.dag,
                        reward=0.0 if self.config.step_reward_signal == "delta_log_reward" else (cur.reward if self.config.reward_every_step else 0.0),
                        z=cur.z,
                        op=None,
                        log_pf=_log_prob(cur.stop_prob),
                        log_pb=0.0,
                        stop_prob=cur.stop_prob,
                    )
                )
                break

            op = cur.edge_ops[action_index]
            next_dag = apply_op(cur.dag, op)
            use_true_next = evaluator is not None and self._should_true_eval(
                step_id=step + 1,
                eval_calls=eval_calls,
                force_terminal=False,
            )
            nxt = self._build_state_context(
                next_dag,
                agent_vectors,
                prev_z=cur.z,
                evaluator=evaluator,
                use_true_eval=use_true_next,
                question_text=question_text,
                question_vector=question_vector,
            )
            if use_true_next:
                eval_calls += 1
            terminal_ctx = nxt

            # Backward policy PB: uniform over all parent states under the same grammar.
            parent_count = self._parent_state_count(nxt.dag)
            log_pb = -math.log(parent_count)

            transition = _TransitionRecord(
                step_id=step,
                cur=cur,
                nxt=nxt,
                action_index=action_index,
                op=op,
                log_pf_action=_log_prob(cur.probs[action_index]),
                log_pb=log_pb,
            )
            transitions.append(transition)

            trajectory.append(
                TrajectoryStep(
                    step_id=step,
                    action=op_to_str(op, cur.dag.nodes),
                    graph=nxt.dag,
                    reward=(
                        (nxt.log_reward - cur.log_reward)
                        if self.config.step_reward_signal == "delta_log_reward"
                        else (cur.reward if self.config.reward_every_step else 0.0)
                    ),
                    z=cur.z,
                    op=op,
                    log_pf=transition.log_pf_action,
                    log_pb=transition.log_pb,
                    stop_prob=cur.stop_prob,
                )
            )

            dag = nxt.dag
            prev_z = cur.z

        if terminal_ctx is None:
            terminal_ctx = self._build_state_context(
                dag,
                agent_vectors,
                prev_z=prev_z,
                evaluator=evaluator,
                use_true_eval=evaluator is not None,
                question_text=question_text,
                question_vector=question_vector,
            )

        # Ensure terminal state receives true evaluation at least once if requested.
        if evaluator is not None and self.config.true_eval_terminal_always:
            terminal_ctx = self._build_state_context(
                terminal_ctx.dag,
                agent_vectors,
                prev_z=terminal_ctx.z,
                evaluator=evaluator,
                use_true_eval=True,
                question_text=question_text,
                question_vector=question_vector,
            )

        if trajectory and (not self.config.reward_every_step):
            trajectory[-1].reward = terminal_ctx.reward

        sampled = SampledDAG(
            graph=terminal_ctx.dag,
            z=terminal_ctx.z,
            reward=terminal_ctx.reward,
            trajectory=trajectory,
            reward_breakdown=terminal_ctx.reward_breakdown,
            task_tag=task_tag,
        )
        return sampled, transitions

    def sample_one(
        self,
        nodes: List[str],
        agent_vectors: Dict[str, Vector],
        evaluator: Optional[MASTaskEvaluator] = None,
        question_text: Optional[str] = None,
        question_vector: Optional[Vector] = None,
        task_tag: Optional[str] = None,
    ) -> SampledDAG:
        sampled, _ = self._sample_one_with_records(
            nodes,
            agent_vectors,
            evaluator=evaluator,
            question_text=question_text,
            question_vector=question_vector,
            task_tag=task_tag,
        )
        return sampled

    def sample_batch(
        self,
        nodes: List[str],
        agent_vectors: Dict[str, Vector],
        num_dags: int,
        evaluator: Optional[MASTaskEvaluator] = None,
        question_text: Optional[str] = None,
        question_vector: Optional[Vector] = None,
        task_tag: Optional[str] = None,
    ) -> List[SampledDAG]:
        return [
            self.sample_one(
                nodes,
                agent_vectors,
                evaluator=evaluator,
                question_text=question_text,
                question_vector=question_vector,
                task_tag=task_tag,
            )
            for _ in range(num_dags)
        ]

    def _policy_outputs_torch(self, ctx: _StateContext) -> Tuple[torch.Tensor, torch.Tensor]:
        edge_scores, z_t, q_t = self._policy_tensors(
            ctx.z, ctx.question_vector, ctx.src_vecs, ctx.dst_vecs, ctx.edge_ops
        )
        if edge_scores.numel() == 0:
            stop_prob = torch.tensor(1.0)
            edge_probs = edge_scores
            return edge_probs, stop_prob
        edge_probs = F.softmax(edge_scores, dim=0)
        stop_prob = torch.sigmoid(self.policy.stop_logit(z_t, q_t))
        return edge_probs, stop_prob

    def _db_loss(self, transitions: List[_TransitionRecord]) -> torch.Tensor:
        if not transitions:
            return torch.tensor(0.0)
        loss = torch.tensor(0.0)
        for tr in transitions:
            edge_probs_cur, stop_prob_cur = self._policy_outputs_torch(tr.cur)
            _, stop_prob_nxt = self._policy_outputs_torch(tr.nxt)
            if edge_probs_cur.numel() == 0:
                continue
            log_pf_action = torch.log((1.0 - stop_prob_cur).clamp(min=1e-12)) + torch.log(
                edge_probs_cur[tr.action_index].clamp(min=1e-12)
            )
            log_stop_cur = torch.log(stop_prob_cur.clamp(min=1e-12))
            log_stop_nxt = torch.log(stop_prob_nxt.clamp(min=1e-12))
            log_reward_cur = torch.log(torch.tensor(max(tr.cur.reward, 1e-12)))
            log_reward_nxt = torch.log(torch.tensor(max(tr.nxt.reward, 1e-12)))
            delta = log_reward_nxt + tr.log_pb + log_stop_cur - log_reward_cur - log_pf_action - log_stop_nxt
            loss = loss + delta.pow(2)
        return loss / float(len(transitions))

    def _contrastive_loss(self, transitions: List[_TransitionRecord]) -> torch.Tensor:
        if not transitions or self.config.loss_cl_weight <= 0.0:
            return torch.tensor(0.0)
        anchors = torch.tensor([tr.cur.z for tr in transitions], dtype=torch.float32)
        positives = torch.tensor([tr.nxt.z for tr in transitions], dtype=torch.float32)
        n = anchors.size(0)
        if n == 0:
            return torch.tensor(0.0)
        temp = max(1e-6, self.config.cl_temperature)
        h_a = self.proj(anchors)
        h_p = self.proj(positives)
        logits = h_a @ h_p.t() / temp
        labels = torch.arange(n)
        loss = F.cross_entropy(logits, labels)
        ctx_weight = max(0.0, self.config.cl_policy_context_weight)
        if ctx_weight > 0.0:
            ctx_a = torch.stack(
                [
                    self.policy.context_embed(self._tensor(tr.cur.z), self._tensor(tr.cur.question_vector))
                    for tr in transitions
                ],
                dim=0,
            )
            ctx_p = torch.stack(
                [
                    self.policy.context_embed(self._tensor(tr.nxt.z), self._tensor(tr.nxt.question_vector))
                    for tr in transitions
                ],
                dim=0,
            )
            ctx_logits = ctx_a @ ctx_p.t() / temp
            ctx_loss = F.cross_entropy(ctx_logits, labels)
            loss = loss + ctx_weight * ctx_loss
        return loss

    def train(
        self,
        nodes: List[str],
        agent_vectors: Dict[str, Vector],
        evaluator: Optional[MASTaskEvaluator] = None,
        question_text: Optional[str] = None,
        question_vector: Optional[Vector] = None,
        task_tag: Optional[str] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> List[GFlowNetTrainingStats]:
        if not self._initialized:
            self.initialize()

        epochs = epochs if epochs is not None else self.config.gflownet_train_epochs
        batch_size = batch_size if batch_size is not None else self.config.gflownet_batch_size

        history: List[GFlowNetTrainingStats] = []
        for ep in range(1, epochs + 1):
            transitions: List[_TransitionRecord] = []
            batch_samples: List[SampledDAG] = []
            for _ in range(batch_size):
                sampled, recs = self._sample_one_with_records(
                    nodes,
                    agent_vectors,
                    evaluator=evaluator,
                    question_text=question_text,
                    question_vector=question_vector,
                    task_tag=task_tag,
                )
                batch_samples.append(sampled)
                transitions.extend(recs)

            db_loss_t = self._db_loss(transitions)
            cl_loss_t = self._contrastive_loss(transitions)
            total_loss_t = self.config.loss_db_weight * db_loss_t + self.config.loss_cl_weight * cl_loss_t
            if self.optimizer is not None and total_loss_t.requires_grad:
                self.optimizer.zero_grad()
                total_loss_t.backward()
                self.optimizer.step()

            mean_reward = 0.0
            mean_total_score = 0.0
            mean_task_score = 0.0
            mean_success = 0.0
            if batch_samples:
                mean_reward = sum(s.reward for s in batch_samples) / len(batch_samples)
                score_vals = [
                    s.reward_breakdown.total_score
                    for s in batch_samples
                    if s.reward_breakdown is not None
                ]
                if score_vals:
                    mean_total_score = sum(score_vals) / len(score_vals)
                task_scores = [
                    s.reward_breakdown.task_score
                    for s in batch_samples
                    if s.reward_breakdown is not None
                ]
                if task_scores:
                    mean_task_score = sum(task_scores) / len(task_scores)
                success_vals = [
                    s.reward_breakdown.task_success
                    for s in batch_samples
                    if s.reward_breakdown is not None
                ]
                if success_vals:
                    mean_success = sum(success_vals) / len(success_vals)

            db_loss = float(db_loss_t.item())
            cl_loss = float(cl_loss_t.item())
            total_loss = float(total_loss_t.item())
            history.append(
                GFlowNetTrainingStats(
                    epoch=ep,
                    db_loss=db_loss,
                    contrastive_loss=cl_loss,
                    total_loss=total_loss,
                    mean_terminal_reward=mean_reward,
                    mean_total_score=mean_total_score,
                    mean_task_score=mean_task_score,
                    mean_success=mean_success,
                    task_tag=task_tag,
                )
            )

        return history
