from dataclasses import dataclass

@dataclass
class MBPOGeneralHyperparameters:
    num_epochs: int
    nb_steps_per_epoch: int
    eval_interval: int

@dataclass
class ReplayBufferConfigs:
    add_batch_size: int
    sample_batch_size: int
    sample_sequence_length: int
    period: int
    min_length_time_axis: int
    max_length_time_axis: int
    add_sequence_length: int

@dataclass
class DynamicsModelConfigs:
    learning_rate: int
    num_models: int
    hidden_dim: int
    state_dim: int
    action_dim: int
    nb_model_rollouts: int

@dataclass
class PolicyConfigs:
    num_rollouts: int
    rollout_length: int
    batch_size: int
