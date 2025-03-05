"""
Train a simple MLP model using pure JAX to predict heightmap values from coordinates.
"""

import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1'

import jax
import jax.numpy as jnp
from jax import random, grad, jit
import optax
import numpy as np
import matplotlib.pyplot as plt
from simple_heightmap import simple_heightmap_generator

# MLP Model Definition
def init_mlp_params(layer_sizes, rng):
    """Initialize MLP parameters."""
    params = []
    for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
        w_key, b_key = random.split(rng)
        rng, _ = random.split(rng)
        weights = random.normal(w_key, (in_size, out_size)) * jnp.sqrt(2 / in_size)
        biases = random.normal(b_key, (out_size,))
        params.append((weights, biases))
    return params

def mlp_forward(params, x):
    """Forward pass through the MLP."""
    activation = x
    for (weights, biases) in params[:-1]:
        activation = jnp.tanh(jnp.dot(activation, weights) + biases)
    final_weights, final_biases = params[-1]
    return jnp.dot(activation, final_weights) + final_biases

# Loss Function
def mse_loss(params, x, y):
    """Compute mean squared error loss."""
    predictions = mlp_forward(params, x)
    return jnp.mean((predictions - y) ** 2)

# Data Generation
def generate_training_data(rng, num_samples=10000, output_size=(256, 256)):
    """Generate synthetic training data using simple_heightmap_generator."""
    rng, heightmap_rng = random.split(rng)
    heightmap = simple_heightmap_generator(heightmap_rng, output_size=output_size, latent_dim=32)
    height, width = output_size
    # Generate random coordinates
    x_coords = random.uniform(rng, (num_samples,), minval=-2.0, maxval=2.0)
    y_coords = random.uniform(rng, (num_samples,), minval=-2.0, maxval=2.0)
    coords = jnp.stack([x_coords, y_coords], axis=1)
    # Compute corresponding height values
    x_indices = ((x_coords + 2.0) / 4.0 * (width - 1)).astype(jnp.int32)
    y_indices = ((y_coords + 2.0) / 4.0 * (height - 1)).astype(jnp.int32)
    heights = heightmap[y_indices, x_indices]
    return coords, heights

# Training Step
def create_update_step(optimizer):
    @jit
    def update_step(params, x, y, opt_state):
        """Perform one optimization step."""
        loss, grads = jax.value_and_grad(mse_loss)(params, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    
    return update_step

def train_mlp(rng, num_epochs=1000, batch_size=128, learning_rate=1e-3):
    """Train the MLP model."""
    # Initialize model
    layer_sizes = [2, 64, 64, 64, 1]  # 3 hidden layers
    rng, init_rng = random.split(rng)
    params = init_mlp_params(layer_sizes, init_rng)
    
    # Initialize optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    
    # Create update step function with the optimizer
    update_step = create_update_step(optimizer)
    
    # Generate training data
    rng, data_rng = random.split(rng)
    coords, heights = generate_training_data(data_rng, num_samples=10000)
    
    # Training loop
    num_batches = len(coords) // batch_size
    for epoch in range(num_epochs):
        rng, batch_rng = random.split(rng)
        perm = random.permutation(batch_rng, len(coords))
        coords = coords[perm]
        heights = heights[perm]
        epoch_loss = 0.0
        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            batch_x = coords[start:end]
            # Reshape heights to have shape (batch_size, 1) for proper loss calculation
            batch_y = heights[start:end].reshape(-1, 1)
            params, opt_state, loss = update_step(params, batch_x, batch_y, opt_state)
            epoch_loss += loss
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}, Loss: {epoch_loss / num_batches}")
    
    return params

def save_model_params(params, filename="model_params.npz"):
    """Save model parameters to a file."""
    flat_params = {f"layer_{i}_{k}": v for i, layer in enumerate(params) 
                   for k, v in enumerate(layer)}
    np.savez(filename, **flat_params)

def main():
    rng = random.PRNGKey(42)
    params = train_mlp(rng, num_epochs=1000)
    save_model_params(params)
    print("Model trained and parameters saved to model_params.npz")

if __name__ == "__main__":
    main()