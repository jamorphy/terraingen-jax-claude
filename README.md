# terraingen-jax-claude

A simple terrain generation system using JAX for high-performance noise functions. This project demonstrates how to create procedural heightmaps using JAX's array operations and random number generation.

## Features

- Generate procedural heightmaps using JAX
- Cross-platform compatibility (CPU/GPU/TPU)
- Export models to ONNX format
- Simple visualization with Matplotlib

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/jamorphy/terraingen-jax-claude.git
cd terraingen-jax-claude
pip install -r requirements.txt
```

## Usage

Generate a simple heightmap:

```bash
python simple_heightmap.py
```

This will create a `simple_heightmap.png` file in the current directory.

## Tests

Run the tests with pytest:

```bash
pytest
```
