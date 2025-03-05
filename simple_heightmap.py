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
        
    Raises:
        ValueError: If output_size dimensions are not positive
    """
    height, width = output_size
    if height <= 0 or width <= 0:
        raise ValueError("Output size dimensions must be positive")
        
    # Generate a random latent vector
    z = random.normal(rng, (latent_dim,))
    
    # Create a grid of coordinates
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
    
def save_multiple_heightmaps(heightmaps, filename, seeds=None):
    """Save multiple heightmaps as a single image in a grid layout.
    
    Args:
        heightmaps: List of heightmap arrays
        filename: Output filename
        seeds: List of seeds used to generate the heightmaps
        
    Raises:
        ValueError: If heightmaps list is empty
    """
    if not heightmaps:
        raise ValueError("Heightmaps list cannot be empty")
        
    num_maps = len(heightmaps)
    rows = int(np.ceil(np.sqrt(num_maps)))
    cols = int(np.ceil(num_maps / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 16))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    
    for i, heightmap in enumerate(heightmaps):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        im = ax.imshow(heightmap, cmap='terrain')
        if seeds:
            ax.set_title(f"Seed: {seeds[i]}")
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide any unused subplots
    for i in range(num_maps, rows * cols):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.axis('off')
    
    # Add a colorbar
    plt.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, label='Height')
    
    plt.savefig(filename, dpi=150)
    plt.close()

def main():
    # Model parameters (simplified)
    output_size = (256, 256)
    latent_dim = 32
    num_samples = 10  # Generate 10 different heightmaps
    
    print(f"Generating {num_samples} heightmaps with size {output_size}")
    
    heightmaps = []
    seeds = []
    
    for i in range(num_samples):
        # Use different seed for each heightmap
        seed = 42 + i
        seeds.append(seed)
        np_seed = np.array([seed], dtype=np.uint32)
        
        # Create PRNG key (simplified)
        try:
            rng = random.PRNGKey(np_seed[0])
            
            # Generate heightmap
            rng, gen_rng = random.split(rng)
            heightmap = simple_heightmap_generator(gen_rng, output_size=output_size, latent_dim=latent_dim)
            heightmaps.append(np.array(heightmap))  # Convert JAX array to numpy array
            print(f"Generated heightmap {i+1}/{num_samples} with shape: {heightmap.shape}")
        
        except Exception as e:
            print(f"Error during heightmap generation {i+1}: {e}")
            print("Generating a simple random heightmap instead...")
            simple_heightmap = np.random.random(output_size)
            heightmaps.append(simple_heightmap)
            print(f"Added fallback heightmap {i+1}")
    
    # Save all heightmaps in a single image
    print("Saving all heightmaps to a single image...")
    save_multiple_heightmaps(heightmaps, "multiple_heightmaps.png", seeds)
    print(f"Saved {len(heightmaps)} heightmaps to multiple_heightmaps.png")

if __name__ == "__main__":
    main()
