"""
Reference:
- Janner et al. (2019): "When to Trust Your Model: Model-Based Policy Optimization".
https://arxiv.org/abs/1906.08253
"""
import numpy as np
import dm_control
import jax
import jax.numpy as jnp
import flax
import flax.nnx as nnx
import flashbax as fbx
import optax
import functools

from dm_control import suite
from dm_control.rl.control import Environment
from dm_control.viewer import application
from flax.training import train_state
from typing import Tuple, Dict
from dataclasses import dataclass

# TODO: initialize configs with dataclasses

def make_env(
    rng: jax.random.PRNGKey
) -> Environment:
    """
    """
    return suite.load(
        domain_name="acrobot",
        task_name="swingup",
        task_kwargs={'random': rng}
    )

def aggregate_state(state):
    return jnp.concatenate([state['orientations'], state['velocity']])

def get_observation_dim(env):
    observation_spec = env.observation_spec()
    total_dim = sum(
        np.prod(spec.shape) for spec in observation_spec.values()
    )
    return total_dim


@dataclass
class MBPOGeneralHyperparameters:
    num_epochs: int
    nb_steps_per_epoch: int
    model_horizon: int

@dataclass
class ReplayBufferConfigs:
    add_batch_size: int
    sample_batch_size: int
    sample_sequence_length: int
    period: int
    min_length_time_axis: int
    max_length_time_axis: int

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
    learning_rate: int
    hidden_dim: int
    state_dim: int
    action_dim: int
    nb_policy_updates: int

class ModelNetwork(nnx.Module):
    """
    The model is an ensemble of dynamics models that
    outputs parametrize a Gaussian distribution with
    diagonal covariance:
    Piθ (st+1,r | st,at) = N(µiθ(st,at), Σiθ(st,at))).
    """
    def __init__(
        self,
        configs: DynamicsModelConfigs,
        rngs: nnx.Rngs
    ) -> None:
        super(ModelNetwork, self).__init__()

        self.fc1 = nnx.Linear(
            in_features=configs.state_dim+configs.action_dim,
            out_features=configs.hidden_dim,
            kernel_init=nnx.initializers.glorot_uniform(),
            rngs=rngs
        )
        self.fc2 = nnx.Linear(
            in_features=configs.hidden_dim,
            out_features=configs.hidden_dim,
            kernel_init=nnx.initializers.glorot_uniform(),
            rngs=rngs
        )
        self.fc3 = nnx.Linear(
            in_features=configs.hidden_dim,
            out_features=configs.hidden_dim,
            kernel_init=nnx.initializers.glorot_uniform(),
            rngs=rngs
        )
        self.fc4 = nnx.Linear(
            in_features=configs.hidden_dim,
            out_features=configs.hidden_dim,
            kernel_init=nnx.initializers.glorot_uniform(),
            rngs=rngs
        )
        # output size is state_dim + 1 (for reward)
        self.fc_mu = nnx.Linear(
            in_features=configs.hidden_dim,
            out_features=configs.state_dim + 1,
            kernel_init=nnx.initializers.glorot_uniform(),
            rngs=rngs
        )
        self.fc_log_sigma = nnx.Linear(
            in_features=configs.hidden_dim,
            out_features=configs.state_dim + 1,
            kernel_init=nnx.initializers.glorot_uniform(),
            rngs=rngs
        )

    def __call__(
        self,
        inputs: Tuple
    ) -> jax.Array:
        state, action = inputs
        expand_state = jnp.expand_dims(state, axis=0)
        expand_action = jnp.expand_dims(action, axis=0)

        output = jnp.concatenate([expand_state, expand_action], axis=-1)
        output = nnx.relu(self.fc1(output))
        output = nnx.relu(self.fc2(output))
        output = nnx.relu(self.fc3(output))
        output = nnx.relu(self.fc4(output))

        mu = self.fc_mu(output)
        log_sigma = self.fc_log_sigma(output)
        return mu, log_sigma

    def sample(
        self,
        inputs: Tuple,
        rng: jax.random.PRNGKey
    ) -> jax.Array:
        mu, log_sigma = self(inputs)
        sigma = jnp.exp(log_sigma)
        return mu + sigma*jax.random.normal(rng, mu.shape)

class EnsembleModels(nnx.Module):
    def __init__(
        self,
        configs: DynamicsModelConfigs,
        rngs: nnx.Rngs
    ) -> None:
        super(EnsembleModels, self).__init__()
        self.configs = configs
        self.models = np.array([
            ModelNetwork(configs, rngs) for _ in range(configs.num_models)
        ])

    @property
    def get_models(self):
        return self.models

    def __call__(
        self,
        inputs: Tuple[jnp.ndarray, jnp.ndarray],
    ) -> jax.Array:
        """
        Apply each model in the ensemble to the same inputs.
        """
        # state, action = inputs
        # outputs = self._vmap_apply_model(self.models, jnp.array([state, action]))
        # print(f"{outputs = }")
        outputs = []
        for model in self.models:
            outputs.append(model(inputs))
        return outputs

def gaussian_nll(
    mu,
    log_sigma,
    targets
):
    """
    Negative log-likelihood loss for a Gaussian distribution.
    """
    print(f"{targets = }")
    print(f"{mu = }")
    concat_targets = jnp.concatenate([targets[0], targets[1]], axis=1)
    print(f"{concat_targets = }")
    sigma = jnp.exp(log_sigma)
    nll = 0.5*jnp.sum(jnp.square((concat_targets-mu)/sigma) + 2*log_sigma)
    return nll

def loss_fn(
    model: EnsembleModels,
    batch,
):
    """
    Compute the loss function.
    """
    state = batch['state']
    action = batch['action']
    outputs = model((state, action))

    mus, log_sigmas = zip(*outputs)

    mus = jnp.stack(mus)
    log_sigmas = jnp.stack(log_sigmas)

    nll = gaussian_nll(mus, log_sigmas, (batch['next_state'], batch['reward']))

    return nll

@nnx.jit
def train_step(
    model: ModelNetwork,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    batch
):
    """
    Train for a single step.
    """
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch['label'])
    optimizer.update(grads)

class NetworkPolicy(nnx.Module):
    def __init__(
        self,
        configs: PolicyConfigs,
        rngs: nnx.Rngs
    ) -> None:
        super(NetworkPolicy, self).__init__()
        self.configs = configs
        self.fc1 = nnx.Linear(
            in_features=configs.state_dim,
            out_features=configs.hidden_dim,
            kernel_init=nnx.initializers.glorot_uniform(),
            rngs=rngs
        )
        self.fc2 = nnx.Linear(
            in_features=configs.hidden_dim,
            out_features=configs.hidden_dim,
            kernel_init=nnx.initializers.glorot_uniform(),
            rngs=rng
        )
        self.fc3 = nnx.Linear(
            in_features=configs.hidden_dim,
            out_features=configs.action_dim,
            kernel_init=nnx.initializers.glorot_uniform(),
            rngs=rng
        )

    @property
    def get_action_dim(self):
        return self.fc3.out_features

    def __call__(
        self,
        input: jax.Array
    ) -> jax.Array:
        output = nnx.relu(self.fc1(input))
        output = nnx.relu(self.fc2(output))
        output = nnx.softmax(self.fc3(output))
        return output

    def sample_action(
        self,
        input: jax.Array,
        key
    ):
        key, subkey = jax.random.split(jax.random.PRNGKey(key))
        return key, jax.random.categorical(subkey, self(input))

def first_fill_env_replay_buffer(replay_buffer, env: Environment):
    time_step = env.reset()
    action_spec = env.action_spec()
    while not replay_buffer.can_sample():
        action = np.random.uniform(
            action_spec.minimum, action_spec.maximum, size=action_spec.shape
        )
        next_time_step = env.step(action)
        data = {
            "state": aggregate_state(time_step.observation),
            "action": jnp.array(action),
            "reward": jnp.array([env.task.get_reward(env.physics)]),
            "next_state": aggregate_state(next_time_step.observation)
        }
        rb_env_state = replay_buffer.add(
            rb_env_state,
            data
        )
        time_step = next_time_step
    time_step = env.reset()
    return rb_env_state

class MBPOAgent:
    def __init__(
        self,
        env: Environment,
        dynamics_configs: DynamicsModelConfigs,
        policy_configs: PolicyConfigs,
        replay_buffer_configs: ReplayBufferConfigs,
        rngs: nnx.Rngs
    ):
        """
        """
        # D_env
        self.replay_buffer_env = fbx.make_trajectory_buffer(
            add_batch_size=replay_buffer_configs.add_batch_size,
            sample_batch_size=replay_buffer_configs.sample_batch_size,
            sample_sequence_length=replay_buffer_configs.sample_sequence_length,
            period=replay_buffer_configs.period,
            min_length_time_axis=replay_buffer_configs.min_length_time_axis,
            max_length_time_axis=replay_buffer_configs.max_length_time_axis,
        )
        dummy_data = {
            "state": jnp.array([1, 1, 1, 1, 0., 0.]),
            "action": jnp.array([0]),
            "reward": jnp.array([1]),
            "next_state": jnp.array([1, 1, 1, 1, 0., 0.])
        }
        self.rb_env_state = self.replay_buffer_env.init(dummy_data)
        self.rb_env_state = first_fill_env_replay_buffer(self.replay_buffer_env, env)

        # D_model
        self.replay_buffer_model = fbx.make_trajectory_buffer(
            add_batch_size=replay_buffer_configs.add_batch_size,
            sample_batch_size=replay_buffer_configs.sample_batch_size,
            sample_sequence_length=replay_buffer_configs.sample_sequence_length,
            period=replay_buffer_configs.period,
            min_length_time_axis=replay_buffer_configs.min_length_time_axis,
            max_length_time_axis=replay_buffer_configs.max_length_time_axis,
        )
        self.rb_model_state = self.replay_buffer_model.init(dummy_data)

        self.policy_configs = policy_configs
        self.policy = NetworkPolicy(policy_configs, rngs)

        self.model_configs = dynamics_configs
        self.model = EnsembleModels(dynamics_configs, rngs)

        self.optimizer_model = optax.adam(learning_rate=dynamics_configs.learning_rate)
        self.optimizer_policy = optax.adam(learning_rate=policy_configs.learning_rate)

    def simulate(self, state, action):
        """
        """
        pass

    def act(self, observation):
        """
        """
        pass

    def update_model(self):
        """
        """
        pass

    def update_policy(self):
        """
        """
        pass

    def train(self):
        """
        """
        pass

    def save(self, file_path: str):
        """
        """
        pass

    def load(self, file_path: str):
        """
        """
        pass


def mbpo_loop(
    agent: MBPOAgent,
    general_configs: MBPOGeneralHyperparameters,
):
    """
    """
    pass

def save_model_params(state: train_state.TrainState, file_path: str):
    """
    Save model parameters.
    """
    params = flax.jax_utils.unreplicate(state.params)
    np.save(file_path, params)
    print(f"Model parameters saved to {file_path}")

def load_model_params(file_path: str):
    """
    Load model parameters.
    """
    params = np.load(file_path, allow_pickle=True)
    return params

def evaluate_agent(agent, env, num_episodes=5):
    pass

def plan_action(timestep, policy_fn):
    """
    """
    action = policy_fn(timestep.observation)
    return action

if __name__ == "__main__":
    seed = 33
    rng = jax.random.PRNGKey(seed)
    env = make_env(rng)
    time_step = env.reset()

    state = time_step.observation
    action_spec = env.action_spec()
    print(f"{type(env) = }")

    # Example action (assuming action space is [-1, 1])
    action = np.random.uniform(
        action_spec.minimum, action_spec.maximum, size=action_spec.shape
    )

    # Dynamics model configs
    dynamics_configs = DynamicsModelConfigs(
        learning_rate=1e-3,
        num_models=5,
        hidden_dim=64,
        state_dim=get_observation_dim(env),
        action_dim=action_spec.shape[0],
        nb_model_rollouts=100
    )

    # Initialize ensemble models
    ensemble = EnsembleModels(dynamics_configs, nnx.Rngs(seed))
    print(f"Ensemble models")

    print(jnp.array(action))

    batch = {
        'state': aggregate_state(state),
        'action': jnp.array(action),
        'reward': jnp.array([env.task.get_reward(env.physics)]),
        'next_state': aggregate_state(state),
    }

    print(f"{batch = }")
    print(f"{batch['state'] = }")
    print(f"{batch['action'] = }")
    print(f"{batch['reward'] = }")
    print(f"{batch['next_state'] = }")
    print(f"batch created")

    # Call the model and compute loss
    outputs = ensemble((batch['state'], batch['action']))
    print("Model outputs:", outputs)

    # Compute loss
    nll_loss = loss_fn(ensemble, batch)
    print("Negative Log-Likelihood Loss:", nll_loss)

    replay_buffer_configs = ReplayBufferConfigs(
        add_batch_size=1,
        sample_batch_size=1,
        sample_sequence_length=1,
        period=1,
        min_length_time_axis=1
    )

    replay_buffer = fbx.make_trajectory_buffer(
        replay_buffer_configs
    )
    print(replay_buffer.buffer)
    print()
    print()
    print()
    replay_buffer.add_experience(
        aggregate_state(state),
        jnp.array([action]),
        aggregate_state(state),
        jnp.array([[env.task.get_reward(env.physics)]]),
        False
    )
    print(replay_buffer.buffer)

# if __name__ == "__main__":

#     seed = 0
#     key = jax.random.PRNGKey(seed)
#     env = make_env(key)
#     action_spec = env.action_spec()
#     print(action_spec)
#     timestep = env.reset()
#     step_jit = jax.jit(env.step)

#     while True:
#         key, subkey = jax.random.split(key)
#         rnd_action = jax.random.uniform(subkey, shape=action_spec.shape, minval=action_spec.minimum, maxval=action_spec.maximum)
#         timestep = step_jit(rnd_action)
#         pixels = env.physics.render(height=480, width=640)
#         print(f"Reward: {timestep.reward}")

#     # For evaluation
#     # app = application.Application(title="DeepMind Control Suite Test")
#     # app.launch(environment_loader=lambda: env, policy=random_policy)
