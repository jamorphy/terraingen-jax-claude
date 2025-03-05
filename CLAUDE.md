# CLAUDE.md - AI Coding Assistant Guidelines

## Build & Test Commands
- Install dependencies: `pip install -r requirements.txt`
- Generate heightmap: `python simple_heightmap.py`
- Export to ONNX: `python onnx_exporter.py`
- Run inference: `python inference.py --model heightmap_generator.onnx --output terrain.png --seed 42`
- Run tests: `pytest`
- Run single test: `pytest test_heightmap.py::test_function_name`
- Type checking: `mypy .`
- Linting: `flake8 .`

## Repository Management
- Clone repo: `git clone https://github.com/jamorphy/terraingen-jax-claude.git`
- Add files: `git add <filename>`
- Commit changes: `git commit -m "Commit message"`
- Push to GitHub: `git push origin main`

## Code Style Guidelines
- Use Python 3.10+ with type annotations
- Follow PEP 8 style guide with line length up to 100 characters
- Sort imports using isort (stdlib → third-party → local)
- Use explicit error handling with appropriate exception types
- Naming: snake_case for functions/variables, PascalCase for classes
- Use pure JAX for all model implementation (no Flax/Haiku)
- Group ONNX-related operations into dedicated modules
- Document public functions with docstrings (Google style)
- Use residual connections in neural network blocks
- Add type hints for all functions and parameters

## Project Structure
- `simple_heightmap.py`: Core heightmap generation functionality
- `onnx_exporter.py`: Tools for exporting models to ONNX format
- `test_heightmap.py`: Test suite for the heightmap generator
- `requirements.txt`: Project dependencies
