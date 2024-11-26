import numpy as np
import gymnasium as gym
import jax
import jax.numpy as jnp
import flax
import flax.nnx as nnx
import flashbax as fbx
import optax
import matplotlib.pyplot as plt

from sbx import SAC
from flax.training import train_state

from configs import (
    MBPOGeneralHyperparameters,
    ReplayBufferConfigs,
    DynamicsModelConfigs,
    PolicyConfigs
)
from network import ModelNetwork, EnsembleModels


# TODO: initialize configs with dataclasses
# TODO: functions to jit!

def sample_transition(
    mu: jax.Array,
    log_sigma: jax.Array,
    rng: jax.random.PRNGKey
) -> jax.Array:
    """
    Sample a transition from a Gaussian distribution.

    Args:
        - mu: Mean of the Gaussian distribution.
        - log_sigma: Logarithm of the standard deviation of the Gaussian distribution.
        - rng: Random number generator.

    Returns:
        - Transition sampled from the Gaussian distribution.
    """
    sigma = jnp.exp(log_sigma)
    return mu + sigma*jax.random.normal(rng, mu.shape)

def loss_fn(
    model: EnsembleModels,
    batch,
):
    """
    Compute the loss for the model.
    
    Args:
        - model: Ensemble of models.
        - batch: Batch of data.
    
    Returns:
        - Loss value.
    """
    def gaussian_nll(
        mu,
        log_sigma,
        targets
    ):
        concat_targets = jnp.concatenate([targets[0], targets[1]], axis=1)
        sigma = jnp.exp(log_sigma)
        nll = 0.5*jnp.sum(jnp.square((concat_targets-mu)/sigma) + 2*log_sigma)
        return nll

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
    Perform a training step.
    
    Args:
        - model: Model to train.
        - optimizer: Optimizer.
        - metrics: Metrics to track.
        - batch: Batch of data.
    """
    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model, batch)
    optimizer.update(grads)

def first_fill_env_replay_buffer(
    replay_buffer,
    rb_env_state,
    env: gym.Env
):
    """
    Fill the replay buffer with initial data.
    
    Args:
        - replay_buffer: Replay buffer.
        - rb_env_state: Replay buffer state.
        - env: Environment.
    
    Returns:
        - Updated replay buffer state.
    """
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
    """
    Model-Based Policy Optimization Agent.
    """
    def __init__(
        self,
        env: gym.Env,
        dynamics_configs: DynamicsModelConfigs,
        policy_configs: PolicyConfigs,
        replay_buffer_configs: ReplayBufferConfigs,
        rngs: nnx.Rngs
    ):
        """
        Initialize the agent.
        
        Args:
            - env: Environment.
            - dynamics_configs: Dynamics model configurations.
            - policy_configs: Policy configurations.
            - replay_buffer_configs: Replay buffer configurations.
            - rngs: Random number generators.
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
        print(f"{dummy_data = }")
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
        Simulate using the dynamics model.
        
        Args:
            - state: Current state.
            - action: Action.

        Returns:
            - Next state.
            - Reward.
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
        Sample an action from the policy.
        
        Args:
            - state: Current state.
        
        Returns:
            - Action.
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
        Update the dynamics model.
        """
        # TODO: when training check how sampling is done
        self.shared_key, subkey = jax.random.split(self.shared_key)
        sample_data = self.replay_buffer_env.sample(self.rb_env_state, subkey)
        loss = loss_fn(self.model, sample_data)
        train_step(self.model, self.optimizer_model, loss, sample_data)

    def generate_model_rollouts(self):
        """
        Generate model rollouts.
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
        Update the policy.
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
    Evaluate the agent.
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
    Main training loop for MBPO.
    (see pseudocode in the paper: Algorithm 2)
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
                "state": jnp.expand_dims(jnp.expand_dims(state, axis=0), axis=1),
                "action": jnp.expand_dims(jnp.expand_dims(action, axis=0), axis=1),
                "reward": jnp.expand_dims(jnp.expand_dims(reward, axis=0), axis=1),
                "next_state": jnp.expand_dims(jnp.expand_dims(next_state, axis=0), axis=1),
            }
            try:
                agent.rb_env_state = agent.replay_buffer_env.add(agent.rb_env_state, env_data)
            except Exception as e:
                print(f"Error adding to buffer: {e}")
            agent.generate_model_rollouts()
            agent.update_policy()
            if done or truncated:
                state, _ = env.reset()
            else:
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
