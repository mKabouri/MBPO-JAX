"""
Reference:
- Janner et al. (2019): "When to Trust Your Model: Model-Based Policy Optimization".
https://arxiv.org/abs/1906.08253
"""
import os
import numpy as np
import gymnasium as gym
import jax
import jax.numpy as jnp
import flax
import flax.nnx as nnx
import flashbax as fbx
import optax
import matplotlib.pyplot as plt

# from stable_baselines3.sac import SAC
from sbx import SAC
from flax.training import train_state
from typing import Tuple, Dict
from dataclasses import dataclass

# TODO: initialize configs with dataclasses
# TODO: functions to jit!

def make_env() -> gym.Env:
    """
    """
    return gym.make("InvertedPendulum-v5", render_mode="human")

@dataclass
class MBPOGeneralHyperparameters:
    num_epochs: int
    nb_steps_per_epoch: int

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
        if len(action.shape) == 0:
            action = jnp.expand_dims(action, axis=0)

        output = jnp.concatenate([state, action], axis=-1)
        output = nnx.relu(self.fc1(output))
        output = nnx.relu(self.fc2(output))
        output = nnx.relu(self.fc3(output))
        output = nnx.relu(self.fc4(output))

        mu = self.fc_mu(output)
        log_sigma = self.fc_log_sigma(output)
        return mu, log_sigma

def sample_transition(
    mu: jax.Array,
    log_sigma: jax.Array,
    rng: jax.random.PRNGKey
) -> jax.Array:
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
        self.model1 = ModelNetwork(configs, rngs)
        self.model2 = ModelNetwork(configs, rngs)
        self.model3 = ModelNetwork(configs, rngs)
        self.model4 = ModelNetwork(configs, rngs)
        self.model5 = ModelNetwork(configs, rngs)

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
        outputs.append(self.model1(inputs))
        outputs.append(self.model2(inputs))
        outputs.append(self.model3(inputs))
        outputs.append(self.model4(inputs))
        outputs.append(self.model5(inputs))
        assert len(outputs) == self.configs.num_models
        return outputs

def gaussian_nll(
    mu,
    log_sigma,
    targets
):
    """
    Negative log-likelihood loss for a Gaussian distribution.
    """
    concat_targets = jnp.concatenate([targets[0], targets[1]], axis=1)
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
    state = jnp.squeeze(batch.experience['state'], axis=1)
    action = jnp.squeeze(batch.experience['action'], axis=1)
    outputs = model((state, action))
    mus, log_sigmas = zip(*outputs)

    mus = jnp.stack(mus)
    log_sigmas = jnp.stack(log_sigmas)

    next_state = jnp.squeeze(batch.experience['next_state'], axis=1)
    reward = batch.experience['reward']
    nll = gaussian_nll(mus, log_sigmas, (next_state, reward))
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
    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model, batch)
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
            rngs=rngs
        )
        self.fc3 = nnx.Linear(
            in_features=configs.hidden_dim,
            out_features=configs.action_dim,
            kernel_init=nnx.initializers.glorot_uniform(),
            rngs=rngs
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
        key, subkey = jax.random.split(key)
        return key, jax.random.categorical(subkey, self(input))

def first_fill_env_replay_buffer(replay_buffer, rb_env_state, env: gym.Env):
    obs, _ = env.reset()

    while not replay_buffer.can_sample(rb_env_state):
        action = env.action_space.sample()
        new_obs, reward, done, truncated, _ = env.step(action)
        data = {
            "state": jnp.array(obs),
            "action": jnp.array(action),
            "reward": jnp.array(reward),
            "next_state": jnp.array(new_obs)
        }
        rb_env_state = replay_buffer.add(
            rb_env_state,
            data
        )
        if done or truncated:
            obs, _ = env.reset()
        else:
            obs = new_obs
    obs, _ = env.reset()
    return rb_env_state

class MBPOAgent:
    def __init__(
        self,
        env: gym.Env,
        dynamics_configs: DynamicsModelConfigs,
        policy_configs: PolicyConfigs,
        replay_buffer_configs: ReplayBufferConfigs,
        rngs: nnx.Rngs
    ):
        """
        """
        self.shared_key = jax.random.PRNGKey(0)
        self.env = env

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
            "state": jnp.array(env.observation_space.sample()),
            "action": jnp.array(env.action_space.sample()),
            "reward": jnp.array(0),
            "next_state": jnp.array(env.observation_space.sample())
        }
        self.rb_env_state = self.replay_buffer_env.init(dummy_data)
        broadcast_fn = lambda x: jnp.broadcast_to(
            x,
            (
                replay_buffer_configs.add_batch_size,
                replay_buffer_configs.add_sequence_length,
                *x.shape
            )
        )
        fake_batch_sequence = jax.tree.map(broadcast_fn, dummy_data)
        self.rb_env_state = self.replay_buffer_env.add(self.rb_env_state, fake_batch_sequence)

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
        self.sac_policy = SAC(
            policy="MlpPolicy",
            env=env,
            buffer_size=5000,
        )

        self.model_configs = dynamics_configs
        self.model = EnsembleModels(dynamics_configs, rngs)
        self.optimizer_model = nnx.Optimizer(
            self.model,
            optax.adam(learning_rate=dynamics_configs.learning_rate)
        )

    def simulate(self, state, action):
        """
        """
        outputs = self.model((state, action))
        self.shared_key, subkey = jax.random.split(self.shared_key)
        rnd_model_idx = jax.random.randint(subkey, (1,), 0, self.model_configs.num_models)[0]
        mu, log_sigma = outputs[rnd_model_idx]

        self.shared_key, subkey = jax.random.split(self.shared_key)
        transition = sample_transition(mu, log_sigma, subkey)
        next_state, reward = transition[:-1], transition[-1]
        return next_state, reward

    def act(self, state):
        """
        """
        state = jnp.expand_dims(state, axis=0)
        self.shared_key, subkey = jax.random.split(self.shared_key)
        action_distribution = self.sac_policy.actor.apply(
            self.sac_policy.policy.actor_state.params,
            state
        )
        action = action_distribution.sample(seed=subkey)
        return action.squeeze()

    def update_model(self):
        """
        TODO: when training check how sampling is done
        """
        self.shared_key, subkey = jax.random.split(self.shared_key)
        sample_data = self.replay_buffer_env.sample(self.rb_env_state, subkey)
        loss = loss_fn(self.model, sample_data)
        train_step(self.model, self.optimizer_model, loss, sample_data)

    def generate_model_rollouts(self):
        """
        """
        for _ in range(self.policy_configs.num_rollouts):
            self.shared_key, subkey = jax.random.split(self.shared_key)
            sampled_data = self.replay_buffer_env.sample(self.rb_env_state, subkey)
            state = sampled_data.experience['state'][0]
            for _ in range(self.policy_configs.rollout_length):
                action = self.act(state)
                next_state, reward = self.simulate(state[0], action)
                done = False
                self.sac_policy.replay_buffer.add(
                    state[0],
                    next_state,
                    jnp.array([action]),
                    jnp.array([reward]),
                    done,
                    [{}]
                )
                state = next_state

    def update_policy(self):
        """
        """
        gradient_steps = self.policy_configs.num_rollouts*self.policy_configs.rollout_length
        self.sac_policy.train(
            gradient_steps=gradient_steps,
            batch_size=self.policy_configs.batch_size
        )

def evaluate_agent(
    agent: MBPOAgent,
    env: gym.Env,
    num_episodes=5
):
    """
    """
    total_rewards = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            episode_reward += reward
            state = next_state
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}/{num_episodes}: Total Reward = {episode_reward}")
    avg_reward = sum(total_rewards)/num_episodes
    print(f"Average Reward over {num_episodes} episodes: {avg_reward}")
    return avg_reward

def mbpo_loop(
    agent: MBPOAgent,
    general_configs: MBPOGeneralHyperparameters,
):
    """
    """
    env = agent.env
    state, _ = env.reset()
    rewards_per_epoch = []
    for epoch in range(general_configs.num_epochs):
        print(f"Epoch {epoch + 1}/{general_configs.num_epochs}")
        agent.update_model()
        total_reward = 0
        for _ in range(general_configs.nb_steps_per_epoch):
            action = agent.act(state)
            next_state, reward, done, truncated, _ = env.step(jnp.array([action]))
            total_reward += reward
            env_data = {
                "state": jnp.array(state),
                "action": jnp.array([action]),
                "reward": jnp.array(reward),
                "next_state": jnp.array(next_state)
            }
            agent.rb_env_state = agent.replay_buffer_env.add(agent.rb_env_state, env_data)

            agent.generate_model_rollouts()
            agent.update_policy()

            state = next_state

        rewards_per_epoch.append(total_reward)
        print(f"Total Reward for Epoch {epoch + 1}: {total_reward}")
        if (epoch + 1) % general_configs.eval_interval == 0:
            print("Evaluating agent...")
            evaluate_agent(agent, env, num_episodes=5)
    # Plot the rewards
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, general_configs.num_epochs + 1), rewards_per_epoch, label="Total Reward per Epoch")
    plt.xlabel("Epochs")
    plt.ylabel("Total Reward")
    plt.title("Training Progress")
    plt.legend()
    plt.grid()
    plt.show()

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

if __name__ == "__main__":
    seed = 33
    env = make_env()
    obs, _ = env.reset()
    # rng = jax.random.PRNGKey(seed)

    # Configs
    dynamics_configs = DynamicsModelConfigs(
        learning_rate=1e-3,
        num_models=5,
        hidden_dim=64,
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        nb_model_rollouts=100
    )
    policy_configs = PolicyConfigs(
        num_rollouts=40,
        rollout_length=1,
        batch_size=10
    )
    rb_configs = ReplayBufferConfigs(
        add_batch_size=1,
        sample_batch_size=4,
        sample_sequence_length=1,
        period=1,
        min_length_time_axis=16,
        max_length_time_axis=32,
        add_sequence_length=10
    )
    general_configs = MBPOGeneralHyperparameters(
        num_epochs=20,
        nb_steps_per_epoch=1000
    )
    # MBPO agent
    agent = MBPOAgent(
        env,
        dynamics_configs,
        policy_configs,
        rb_configs,
        nnx.Rngs(seed)
    )
    print(f"Agent initialized.")

    mbpo_loop(agent, general_configs)

    env.close()
