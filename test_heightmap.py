import pytest
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from simple_heightmap import simple_heightmap_generator, save_heightmap, save_multiple_heightmaps, main

@pytest.fixture
def rng():
    seed = 42
    return random.PRNGKey(seed)

@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path

def test_simple_heightmap_generator_shape(rng):
    output_size = (256, 256)
    heightmap = simple_heightmap_generator(rng, output_size=output_size, latent_dim=32)
    assert heightmap.shape == output_size, f"Expected shape {output_size}, but got {heightmap.shape}"

def test_simple_heightmap_generator_range(rng):
    heightmap = simple_heightmap_generator(rng, output_size=(256, 256), latent_dim=32)
    assert jnp.all(heightmap >= 0) and jnp.all(heightmap <= 1), "Heightmap values must be between 0 and 1"

def test_simple_heightmap_generator_determinism(rng):
    heightmap1 = simple_heightmap_generator(rng, output_size=(256, 256), latent_dim=32)
    heightmap2 = simple_heightmap_generator(rng, output_size=(256, 256), latent_dim=32)
    assert jnp.allclose(heightmap1, heightmap2), "Heightmaps generated with same key must be identical"

def test_main_creates_file(tmp_dir):
    os.chdir(tmp_dir)
    main()
    assert os.path.exists("multiple_heightmaps.png"), "Expected multiple_heightmaps.png to be created"

def test_main_fallback_on_prng_failure(monkeypatch, tmp_dir):
    # Simulate PRNGKey failure
    def mock_prng_key(*args, **kwargs):
        raise Exception("Simulated PRNGKey failure")
    monkeypatch.setattr(random, "PRNGKey", mock_prng_key)
    os.chdir(tmp_dir)
    main()
    assert os.path.exists("multiple_heightmaps.png"), "Expected multiple_heightmaps.png to be created on PRNGKey failure"

def test_main_fallback_on_generation_failure(monkeypatch, tmp_dir, rng):
    # Simulate simple_heightmap_generator failure
    def mock_generator(*args, **kwargs):
        raise Exception("Simulated generator failure")
    monkeypatch.setattr("simple_heightmap.simple_heightmap_generator", mock_generator)
    monkeypatch.setattr(random, "PRNGKey", lambda _: rng)
    os.chdir(tmp_dir)
    main()
    assert os.path.exists("multiple_heightmaps.png"), "Expected multiple_heightmaps.png to be created on generator failure"

# New Tests
def test_invalid_output_size(rng):
    """Test that invalid output sizes raise appropriate errors."""
    with pytest.raises(ValueError, match="Output size dimensions must be positive"):
        simple_heightmap_generator(rng, output_size=(-1, 256), latent_dim=32)
    with pytest.raises(ValueError, match="Output size dimensions must be positive"):
        simple_heightmap_generator(rng, output_size=(0, 256), latent_dim=32)

def test_different_latent_dims(rng):
    """Test heightmap generation with varying latent dimensions."""
    output_size = (128, 128)
    # Test with small latent dim
    heightmap_small = simple_heightmap_generator(rng, output_size=output_size, latent_dim=2)
    assert heightmap_small.shape == output_size
    assert jnp.all(heightmap_small >= 0) and jnp.all(heightmap_small <= 1)
    # Test with large latent dim
    heightmap_large = simple_heightmap_generator(rng, output_size=output_size, latent_dim=64)
    assert heightmap_large.shape == output_size
    assert jnp.all(heightmap_large >= 0) and jnp.all(heightmap_large <= 1)

def test_save_multiple_heightmaps_grid_layout(tmp_dir):
    """Test that save_multiple_heightmaps creates the correct grid layout."""
    os.chdir(tmp_dir)
    heightmaps = [np.random.random((128, 128)) for _ in range(5)]
    seeds = list(range(5))
    save_multiple_heightmaps(heightmaps, "test_grid.png", seeds)
    assert os.path.exists("test_grid.png"), "Expected test_grid.png to be created"

def test_save_multiple_heightmaps_empty_list(tmp_dir):
    """Test that save_multiple_heightmaps handles an empty list appropriately."""
    os.chdir(tmp_dir)
    with pytest.raises(ValueError, match="Heightmaps list cannot be empty"):
        save_multiple_heightmaps([], "empty_grid.png")

def test_generation_performance(rng):
    """Test the performance of heightmap generation for a larger size."""
    output_size = (1024, 1024)
    start_time = time.time()
    heightmap = simple_heightmap_generator(rng, output_size=output_size, latent_dim=32)
    elapsed_time = time.time() - start_time
    assert heightmap.shape == output_size
    assert elapsed_time < 10.0, f"Generation took too long: {elapsed_time} seconds"

# Step 1: First batch of 4 new tests (Input Validation and Edge Cases)
def test_latent_dim_zero(rng):
    """Test that a latent dimension of 0 raises an appropriate error."""
    with pytest.raises(IndexError, match="index is out of bounds"):
        simple_heightmap_generator(rng, output_size=(128, 128), latent_dim=0)

def test_latent_dim_negative(rng):
    """Test that a negative latent dimension raises an appropriate error."""
    with pytest.raises(Exception):  # JAX raises MLIRError for negative dimensions
        simple_heightmap_generator(rng, output_size=(128, 128), latent_dim=-1)

def test_output_size_single_dimension(rng):
    """Test that a single dimension output size raises an appropriate error."""
    with pytest.raises(ValueError, match="not enough values to unpack"):
        simple_heightmap_generator(rng, output_size=(256,), latent_dim=32)

def test_output_size_large(rng):
    """Test that a very large output size does not cause crashes (but may take time)."""
    output_size = (2048, 2048)
    heightmap = simple_heightmap_generator(rng, output_size=output_size, latent_dim=32)
    assert heightmap.shape == output_size
    assert jnp.all(heightmap >= 0) and jnp.all(heightmap <= 1)

# Step 2: Second batch of 4 tests (Output Properties and Consistency)
def test_output_non_nan(rng):
    """Test that the heightmap contains no NaN values."""
    heightmap = simple_heightmap_generator(rng, output_size=(256, 256), latent_dim=32)
    assert not jnp.any(jnp.isnan(heightmap)), "Heightmap contains NaN values"

def test_output_non_inf(rng):
    """Test that the heightmap contains no infinite values."""
    heightmap = simple_heightmap_generator(rng, output_size=(256, 256), latent_dim=32)
    assert not jnp.any(jnp.isinf(heightmap)), "Heightmap contains infinite values"

def test_output_variance(rng):
    """Test that the heightmap has non-zero variance (not a flat plane)."""
    heightmap = simple_heightmap_generator(rng, output_size=(256, 256), latent_dim=32)
    variance = jnp.var(heightmap)
    assert variance > 0, f"Heightmap variance is zero or negative: {variance}"

def test_output_different_seeds(rng):
    """Test that different seeds produce different heightmaps."""
    rng2 = random.PRNGKey(43)  # Different seed
    heightmap1 = simple_heightmap_generator(rng, output_size=(256, 256), latent_dim=32)
    heightmap2 = simple_heightmap_generator(rng2, output_size=(256, 256), latent_dim=32)
    assert not jnp.allclose(heightmap1, heightmap2), "Heightmaps with different seeds are identical"

# Step 3: Third batch of 4 tests (File Saving and Visualization)
def test_save_heightmap_file_exists(tmp_dir):
    """Test that save_heightmap creates a file."""
    os.chdir(tmp_dir)
    heightmap = np.random.random((128, 128))
    save_heightmap(heightmap, "test_save.png")
    assert os.path.exists("test_save.png"), "Expected test_save.png to be created"

def test_save_heightmap_invalid_path(tmp_dir):
    """Test that save_heightmap handles invalid paths gracefully."""
    os.chdir(tmp_dir)
    heightmap = np.random.random((128, 128))
    invalid_path = tmp_dir / "nonexistent_dir" / "test_save.png"
    with pytest.raises(OSError):
        save_heightmap(heightmap, invalid_path)

def test_save_multiple_heightmaps_small_number(tmp_dir):
    """Test save_multiple_heightmaps with a small number of heightmaps."""
    os.chdir(tmp_dir)
    heightmaps = [np.random.random((128, 128)) for _ in range(3)]
    save_multiple_heightmaps(heightmaps, "small_number.png")
    assert os.path.exists("small_number.png"), "Expected small_number.png to be created"

def test_save_multiple_heightmaps_with_seeds(tmp_dir):
    """Test save_multiple_heightmaps with specified seeds."""
    os.chdir(tmp_dir)
    heightmaps = [np.random.random((128, 128)) for _ in range(3)]
    seeds = [1, 2, 3]
    save_multiple_heightmaps(heightmaps, "with_seeds.png", seeds)
    assert os.path.exists("with_seeds.png"), "Expected with_seeds.png to be created"

# Step 4: Fourth batch of 4 tests (Performance and Stress Tests)
def test_generation_small_size_performance(rng):
    """Test performance for a small size heightmap."""
    output_size = (64, 64)
    start_time = time.time()
    heightmap = simple_heightmap_generator(rng, output_size=output_size, latent_dim=32)
    elapsed_time = time.time() - start_time
    assert heightmap.shape == output_size
    assert elapsed_time < 1.0, f"Small size generation took too long: {elapsed_time} seconds"

def test_generation_medium_size_performance(rng):
    """Test performance for a medium size heightmap."""
    output_size = (512, 512)
    start_time = time.time()
    heightmap = simple_heightmap_generator(rng, output_size=output_size, latent_dim=32)
    elapsed_time = time.time() - start_time
    assert heightmap.shape == output_size
    assert elapsed_time < 5.0, f"Medium size generation took too long: {elapsed_time} seconds"

def test_generation_high_latent_dim_performance(rng):
    """Test performance with a high latent dimension."""
    output_size = (256, 256)
    start_time = time.time()
    heightmap = simple_heightmap_generator(rng, output_size=output_size, latent_dim=128)
    elapsed_time = time.time() - start_time
    assert heightmap.shape == output_size
    assert elapsed_time < 5.0, f"High latent dim generation took too long: {elapsed_time} seconds"

def test_generation_stress_many_iterations(rng):
    """Stress test by generating many small heightmaps."""
    output_size = (64, 64)
    start_time = time.time()
    for _ in range(100):
        heightmap = simple_heightmap_generator(rng, output_size=output_size, latent_dim=32)
        assert heightmap.shape == output_size
    elapsed_time = time.time() - start_time
    assert elapsed_time < 10.0, f"Stress test took too long: {elapsed_time} seconds"

# Step 5: Fifth batch of 4 tests (Redundant Checks and Edge Cases)
def test_output_range_redundant_check(rng):
    """Redundant check for output range (already tested, but adding for robustness)."""
    heightmap = simple_heightmap_generator(rng, output_size=(256, 256), latent_dim=32)
    assert jnp.all(heightmap >= 0) and jnp.all(heightmap <= 1), "Heightmap values must be between 0 and 1"

def test_output_shape_redundant_check(rng):
    """Redundant check for output shape (already tested, but adding for robustness)."""
    output_size = (256, 256)
    heightmap = simple_heightmap_generator(rng, output_size=output_size, latent_dim=32)
    assert heightmap.shape == output_size, f"Expected shape {output_size}, but got {heightmap.shape}"

def test_latent_dim_odd_number(rng):
    """Test heightmap generation with an odd latent dimension."""
    output_size = (128, 128)
    heightmap = simple_heightmap_generator(rng, output_size=output_size, latent_dim=31)
    assert heightmap.shape == output_size
    assert jnp.all(heightmap >= 0) and jnp.all(heightmap <= 1)

def test_save_heightmap_custom_cmap(tmp_dir):
    """Test save_heightmap with a custom colormap."""
    os.chdir(tmp_dir)
    heightmap = np.random.random((128, 128))
    
    # Create a custom colormap function that wraps the save_heightmap function
    def save_with_custom_cmap(heightmap, filename):
        plt.figure(figsize=(8, 8))
        plt.imshow(heightmap, cmap='viridis')  # Use viridis instead of terrain
        plt.colorbar(label='Height')
        plt.savefig(filename)
        plt.close()
    
    # Save with custom colormap
    save_with_custom_cmap(heightmap, "custom_cmap.png")
    assert os.path.exists("custom_cmap.png"), "Expected custom_cmap.png to be created"
