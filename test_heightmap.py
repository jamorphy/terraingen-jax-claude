import pytest
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import matplotlib.pyplot as plt
import os
from simple_heightmap import simple_heightmap_generator, save_heightmap, main

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
    assert os.path.exists("simple_heightmap.png"), "Expected simple_heightmap.png to be created"

def test_main_fallback_on_prng_failure(monkeypatch, tmp_dir):
    # Simulate PRNGKey failure
    def mock_prng_key(*args, **kwargs):
        raise Exception("Simulated PRNGKey failure")
    monkeypatch.setattr(random, "PRNGKey", mock_prng_key)
    os.chdir(tmp_dir)
    main()
    assert os.path.exists("fallback_heightmap.png"), "Expected fallback_heightmap.png on PRNGKey failure"

def test_main_fallback_on_generation_failure(monkeypatch, tmp_dir, rng):
    # Simulate simple_heightmap_generator failure
    def mock_generator(*args, **kwargs):
        raise Exception("Simulated generator failure")
    monkeypatch.setattr("simple_heightmap.simple_heightmap_generator", mock_generator)
    monkeypatch.setattr(random, "PRNGKey", lambda _: rng)
    os.chdir(tmp_dir)
    main()
    assert os.path.exists("fallback_heightmap.png"), "Expected fallback_heightmap.png on generator failure"
