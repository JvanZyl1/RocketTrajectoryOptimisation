import jax.numpy as jnp
from SupervisoryLearnTry.supervisory_learn import Actor, load_model

# Load the model parameters
loaded_params = load_model('SupervisoryLearnTry/supervisory_network.pkl')

print(loaded_params)