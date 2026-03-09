from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

Edge = Tuple[int, int]
Vector = List[float]
GraphOp = Tuple[str, int, int]


@dataclass
class AgentProfile:
    """Agent metadata for MAS initialization."""

    agent_id: str
    role: str
    profile: str
    system_prompt: str
    metadata: Dict[str, str] = field(default_factory=dict)
    vector: Optional[Vector] = None


@dataclass
class DAGState:
    """Discrete DAG state."""

    nodes: Sequence[str]
    edges: List[Edge] = field(default_factory=list)
    z: Optional[Vector] = None
    reward: float = 0.0


@dataclass
class TrajectoryStep:
    step_id: int
    action: str
    graph: DAGState
    reward: float
    z: Vector
    op: Optional[GraphOp] = None
    log_pf: float = 0.0
    log_pb: float = 0.0
    stop_prob: float = 0.0


@dataclass
class SampledDAG:
    graph: DAGState
    z: Vector
    reward: float
    trajectory: List[TrajectoryStep] = field(default_factory=list)
    reward_breakdown: Optional["RewardBreakdown"] = None
    task_tag: Optional[str] = None


@dataclass
class ProxyPair:
    z: Vector
    reward: float
    graph: DAGState


@dataclass
class ClusterSeed:
    cluster_id: int
    center_z: Vector
    member_indices: List[int]


@dataclass
class OptimizedRepresentation:
    cluster_id: int
    start_z: Vector
    optimized_z: Vector
    proxy_score: float


@dataclass
class OptimizationOutput:
    sampled_dags: List[SampledDAG]
    cluster_seeds: List[ClusterSeed]
    top_optimized_representations: List[OptimizedRepresentation]
    z_best_star: Vector
    matched_discrete_dag: DAGState
    refined_best_dag: DAGState
    refined_best_score: float


@dataclass
class TaskEvaluation:
    """Outcome metrics after a full MAS task run for one DAG."""

    task_score: float
    success: float = 0.0
    latency: float = 0.0
    token_cost: float = 0.0
    safety_penalty: float = 0.0
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class RewardBreakdown:
    bic_score: float
    bic_term: float
    task_utility: float
    task_score: float
    task_success: float
    contribution_term: float
    total_score: float
    reward: float
    question_alignment_term: float = 0.0
    size_penalty_term: float = 0.0
    agent_contributions: Dict[str, float] = field(default_factory=dict)
    component_terms: Dict[str, float] = field(default_factory=dict)


@dataclass
class GFlowNetTrainingStats:
    epoch: int
    db_loss: float
    contrastive_loss: float
    total_loss: float
    mean_terminal_reward: float
    mean_total_score: float
    mean_task_score: float = 0.0
    mean_success: float = 0.0
    task_tag: Optional[str] = None


@dataclass
class MASConfig:
    embedding_dim: int = 32
    embedding_api_base: Optional[str] = None
    embedding_model: Optional[str] = None
    embedding_api_key: Optional[str] = None
    embedding_timeout_s: float = 30.0
    gflownet_max_steps: int = 24
    gflownet_stop_prob: float = 0.20
    reward_every_step: bool = True
    allow_backtracking: bool = False
    num_sampled_dags: int = 40
    cluster_count: int = 5
    kmeans_iters: int = 25
    proxy_train_epochs: int = 80
    proxy_lr: float = 0.02
    gradient_steps: int = 25
    gradient_lr: float = 0.08
    top_optimized_count: int = 5
    hc_max_iters: int = 60
    random_seed: int = 7
    # GFlowNet training loop
    gflownet_train_epochs: int = 80
    gflownet_batch_size: int = 16
    gflownet_lr: float = 0.02
    gflownet_weight_decay: float = 1e-4
    # GFlowNet losses
    loss_db_weight: float = 1.0
    loss_cl_weight: float = 0.1
    cl_temperature: float = 0.2
    cl_proj_dim: int = 16
    # Reward composition
    reward_temperature: float = 2.0
    reward_task_weight: float = 1.0
    reward_bic_weight: float = 0.35
    reward_contrib_weight: float = 0.40
    reward_question_weight: float = 0.25
    reward_size_penalty_weight: float = 0.08
    size_penalty_exempt_agents: Tuple[str, ...] = ("algo_designer",)
    size_penalty_free_count: int = 1
    size_penalty_stage1_threshold: int = 2
    size_penalty_stage1_rate: float = 1.0
    size_penalty_stage2_rate: float = 1.8
    reward_clip_min: float = -20.0
    reward_clip_max: float = 20.0
    bic_normalize_scale: float = 1000.0
    reward_true_eval_cache_max_entries: int = 4096
    reward_question_prior_max_entries: int = 1024
    reward_cache_miss_policy: str = "prior_or_proxy"  # "zero" | "proxy_only" | "prior_or_proxy" | "true_utility"
    reward_cache_miss_true_eval_contrib: bool = False
    use_task_score_as_bic: bool = True
    # Utility terms from MAS run results
    utility_task_score_weight: float = 1.0
    utility_success_weight: float = 0.5
    utility_latency_weight: float = 0.05
    utility_token_weight: float = 0.02
    utility_safety_weight: float = 0.30
    # Contribution estimation
    contribution_mode: str = "loo"  # "loo" or "shapley"
    shapley_permutations: int = 24
    # Sparse true-evaluation scheduling (to control evaluator cost)
    true_eval_interval: int = 6
    true_eval_budget_per_trajectory: int = 4
    true_eval_terminal_always: bool = True
    # Fast step-level signal
    step_reward_signal: str = "delta_log_reward"  # "delta_log_reward" | "dense_reward"
    # GFlowNet action grammar for DB consistency
    gflownet_action_space: str = "edge_add"  # "edge_add" | "extended"
    # Refinement objective
    refine_objective: str = "composite"  # "bic" | "composite" | "auto"
    # Task conditioning / subset selection
    enable_task_conditioning: bool = True
    enable_subset_selection: bool = True
    agent_subset_top_k: int = 4
    agent_subset_min_k: int = 2
    gating_relevance_weight: float = 1.0
    gating_complement_weight: float = 0.6
    gating_diversity_weight: float = 0.3
    enable_learnable_gating: bool = True
    gating_learning_rate: float = 0.05
    gating_l2: float = 1e-4
    gating_update_on_train: bool = True
    gating_update_on_run: bool = False
    gating_update_policy: str = "train_only"  # "none" | "train_only" | "train_and_run"
    gating_feedback_metric: str = "mean_total_score"  # "mean_total_score" | "mean_reward" | "topk_mean_total_score"
    gating_feedback_topk_frac: float = 0.3
    gating_feedback_ema: float = 0.2
    gating_feedback_scale: float = 5.0
    gating_target_clip: float = 1.5
    gating_grad_clip: float = 3.0
    gating_first_update_gain: float = 0.5
    gating_state_max_entries: int = 512
    question_signature_hash_decimals: int = 3
    gating_require_true_evaluator_for_update: bool = True
    gating_question_isolated: bool = True
    # Graph representation (GNN + attention)
    repr_message_steps: int = 2
    repr_attention_edge_bias: float = 0.20
    repr_attention_self_bias: float = 0.10
    # Policy q-injection scales
    policy_q_scale: float = 1.0
    policy_pair_weight: float = 1.0
    policy_q_pair_weight: float = 0.35
    policy_hidden_dim: int = 64
    # CL coupling
    cl_policy_context_weight: float = 0.5
    # Proxy model
    proxy_hidden_dim: int = 64
    proxy_rank_weight: float = 0.1
    proxy_rank_margin: float = 0.05
