import jax
import jax.numpy as jnp
import flax.nnx as nnx
import gymnasium as gym

from configs import (
    DynamicsModelConfigs,
    PolicyConfigs,
    ReplayBufferConfigs,
    MBPOGeneralHyperparameters
)
from mbpo_agent import MBPOAgent, mbpo_loop


if __name__ == "__main__":
    seed = 33
    env = gym.make("InvertedPendulum-v5", render_mode="human")
    obs, _ = env.reset()
    # rng = jax.random.PRNGKey(seed)

    # Configs
    dynamics_configs = DynamicsModelConfigs(
        learning_rate=1e-3,
        num_models=6,
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
        add_sequence_length=1
    )
    general_configs = MBPOGeneralHyperparameters(
        num_epochs=20,
        nb_steps_per_epoch=1000,
        eval_interval=3
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