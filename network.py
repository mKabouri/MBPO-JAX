import jax
import jax.numpy as jnp
import flax.nnx as nnx

from configs import DynamicsModelConfigs
from typing import Tuple

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

class EnsembleModels(nnx.Module):
    """
    """
    def __init__(
        self,
        configs: DynamicsModelConfigs,
        rngs: nnx.Rngs
    ) -> None:
        super().__init__()
        self.configs = configs
        self.models = {
            f"model{i+1}": ModelNetwork(configs, rngs) for i in range(configs.num_models)
        }

    @property
    def get_models(self):
        return list(self.models.values())

    def __call__(
        self,
        inputs: Tuple[jnp.ndarray, jnp.ndarray],
    ) -> jax.Array:
        """
        """
        outputs = [
            model(inputs) for model in self.get_models
        ]
        return outputs

