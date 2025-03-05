"""
Simple heightmap generator using JAX that works on macOS with Metal or CPU.
"""

import os
# Force JAX to use CPU explicitly to avoid Metal issues for now
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
# Configure JAX to avoid device-specific issues
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1'

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import matplotlib.pyplot as plt

def simple_heightmap_generator(rng, output_size=(256, 256), latent_dim=32):
    """Generate a simple heightmap using basic noise operations.
    
    Args:
        rng: JAX random number generator key
        output_size: Tuple of (height, width) for the output heightmap
        latent_dim: Dimension of the latent vector
        
    Returns:
        Generated heightmap as a JAX array
    """
    # Generate a random latent vector
    z = random.normal(rng, (latent_dim,))
    
    # Create a grid of coordinates
    height, width = output_size
    x = jnp.linspace(-2.0, 2.0, width)
    y = jnp.linspace(-2.0, 2.0, height)
    X, Y = jnp.meshgrid(x, y)
    
    # Use the latent vector to scale simple noise
    noise = jnp.sin(X * z[0]) + jnp.cos(Y * z[1])
    for i in range(2, latent_dim, 2):
        if i+1 < latent_dim:
            noise += 0.5 * (jnp.sin(X * z[i]) + jnp.cos(Y * z[i+1])) / (i // 2)
    
    # Normalize to [0, 1]
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    return noise

def save_heightmap(heightmap, filename):
    """Save a heightmap as an image file.
    
    Args:
        heightmap: Heightmap array with shape (height, width)
        filename: Output filename
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(heightmap, cmap='terrain')
    plt.colorbar(label='Height')
    plt.savefig(filename)
    plt.close()

def main():
    # Set random seed for reproducibility
    seed = 42
    np_seed = np.array([seed], dtype=np.uint32)
    
    # Create PRNG key (simplified)
    try:
        rng = random.PRNGKey(np_seed[0])
    except Exception as e:
        print(f"Error creating PRNG key: {e}")
        print("Falling back to CPU numpy random generation")
        rng = np.random.default_rng(seed)
        heightmap = np.random.random((256, 256))
        save_heightmap(heightmap, "fallback_heightmap.png")
        return

    # Model parameters (simplified)
    output_size = (256, 256)
    latent_dim = 32

    print(f"Generating simple heightmap with size {output_size}")

    try:
        # Split RNG and generate heightmap
        rng, gen_rng = random.split(rng)
        heightmap = simple_heightmap_generator(gen_rng, output_size=output_size, latent_dim=latent_dim)
        
        # Save the result
        print("Saving heightmap...")
        save_heightmap(heightmap, "simple_heightmap.png")
        print(f"Generated heightmap with shape: {heightmap.shape}")
    
    except Exception as e:
        print(f"Error during heightmap generation: {e}")
        print("Generating a simple random heightmap instead...")
        simple_heightmap = np.random.random(output_size)
        save_heightmap(simple_heightmap, "fallback_heightmap.png")
        print("Simple random heightmap saved to fallback_heightmap.png")

if __name__ == "__main__":
    main()
