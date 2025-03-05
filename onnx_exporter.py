import jax
import jax.numpy as jnp
from jax import random, jit
import numpy as np
from typing import Dict, Tuple, Any
import onnx
import onnxruntime as ort
from pathlib import Path
import tensorflow as tf
import tf2onnx

from heightmap_generator import ModelParams, heightmap_generator, init_heightmap_generator


def create_inference_fn(params: ModelParams, features: int, depth: int, output_size: Tuple[int, int]):
    """Create a jitted inference function with fixed parameters.
    
    Args:
        params: Model parameters
        features: Number of features in hidden layers
        depth: Number of processing blocks
        output_size: Size of the output heightmap
        
    Returns:
        Jitted inference function
    """
    @jit
    def inference_fn(z):
        return heightmap_generator(z, params, features, depth, output_size)
    
    return inference_fn


def convert_jax_to_tf(
    inference_fn,
    input_shape: Tuple[int, ...],
) -> tf.keras.Model:
    """Convert JAX model to TensorFlow for ONNX export.
    
    Args:
        inference_fn: JAX inference function
        input_shape: Shape of the input tensor
        
    Returns:
        TensorFlow model equivalent to the JAX model
    """
    # Create a TensorFlow model wrapper around the JAX function
    class JaxModel(tf.keras.Model):
        def __init__(self, jax_fn):
            super().__init__()
            self.jax_fn = jax_fn
            
        def call(self, inputs):
            # Convert TF tensor to NumPy array to JAX array
            jax_input = jnp.array(inputs.numpy())
            # Run JAX function
            output = self.jax_fn(jax_input)
            # Convert JAX array to NumPy to TF tensor
            return tf.convert_to_tensor(np.array(output))
    
    # Create model
    inputs = tf.keras.Input(shape=input_shape[1:])
    
    # Create a dummy inference to trace the function
    dummy_input = np.ones(input_shape, dtype=np.float32)
    dummy_output = inference_fn(jnp.array(dummy_input))
    
    # Define the model
    model = JaxModel(inference_fn)
    
    # Create concrete function to ensure shape info is preserved
    concrete_func = tf.function(
        lambda x: model(x)
    ).get_concrete_function(
        tf.TensorSpec(input_shape, tf.float32)
    )
    
    return model


def trace_and_convert_to_onnx(
    model_path: str,
    params: ModelParams,
    latent_dim: int = 128,
    features: int = 256,
    depth: int = 4,
    output_size: Tuple[int, int] = (512, 512),
    opset_version: int = 15
) -> None:
    """Trace the JAX model, convert to TensorFlow, and export to ONNX format.
    
    Args:
        model_path: Path to save the ONNX model
        params: Model parameters
        latent_dim: Dimension of the latent vector
        features: Number of features in hidden layers
        depth: Number of processing blocks
        output_size: Size of the output heightmap
        opset_version: ONNX opset version
    """
    # Create inference function with fixed parameters
    inference_fn = create_inference_fn(params, features, depth, output_size)
    
    # Input shape for the model
    input_shape = (1, latent_dim)
    
    try:
        # Convert JAX model to TensorFlow
        tf_model = convert_jax_to_tf(inference_fn, input_shape)
        
        # Convert to ONNX using tf2onnx
        input_signature = [tf.TensorSpec(input_shape, tf.float32)]
        
        # Get the concrete function
        concrete_func = tf.function(lambda x: tf_model(x)).get_concrete_function(*input_signature)
        
        # Convert to ONNX
        onnx_model, _ = tf2onnx.convert.from_function(
            concrete_func,
            input_signature=input_signature,
            opset=opset_version,
            output_path=model_path
        )
        
        print(f"Model saved to {model_path}")
        
    except Exception as e:
        print(f"Error converting to ONNX: {e}")
        return


def verify_onnx_model(
    model_path: str,
    test_input: np.ndarray
) -> bool:
    """Verify that the ONNX model works as expected.
    
    Args:
        model_path: Path to the ONNX model
        test_input: Test input for the model
        
    Returns:
        True if verification passes, False otherwise
    """
    try:
        # Create an ONNX Runtime session
        session = ort.InferenceSession(model_path)
        
        # Get the name of the input
        input_name = session.get_inputs()[0].name
        
        # Ensure input is float32
        test_input = test_input.astype(np.float32)
        
        # Run inference
        onnx_outputs = session.run(None, {input_name: test_input})
        
        # Check output shape
        output_shape = onnx_outputs[0].shape
        expected_shape = (test_input.shape[0], 512, 512)
        
        if output_shape != expected_shape:
            print(f"Output shape mismatch: got {output_shape}, expected {expected_shape}")
            return False
            
        # Check output range (should be between 0 and 1 for a heightmap)
        if np.min(onnx_outputs[0]) < 0 or np.max(onnx_outputs[0]) > 1:
            print(f"Output range issue: min={np.min(onnx_outputs[0])}, max={np.max(onnx_outputs[0])}")
            return False
            
        print("ONNX model verification passed!")
        return True
        
    except Exception as e:
        print(f"Error verifying ONNX model: {e}")
        return False


def export_model_to_onnx(seed: int = 42) -> None:
    """Create, initialize, and export the heightmap generator model to ONNX.
    
    Args:
        seed: Random seed for reproducibility
    """
    # Set random seed
    rng = random.PRNGKey(seed)
    
    # Model hyperparameters
    latent_dim = 128
    features = 256
    depth = 4
    output_size = (512, 512)
    
    # Initialize the model parameters
    rng, init_rng = random.split(rng)
    params = init_heightmap_generator(
        init_rng, 
        latent_dim=latent_dim,
        features=features,
        depth=depth,
        output_size=output_size
    )
    
    # Export to ONNX
    model_path = "heightmap_generator.onnx"
    
    trace_and_convert_to_onnx(
        model_path=model_path,
        params=params,
        latent_dim=latent_dim,
        features=features,
        depth=depth,
        output_size=output_size
    )
    
    # Verify the ONNX model
    rng, test_rng = random.split(rng)
    test_input = random.normal(test_rng, (1, latent_dim))
    test_input_np = np.array(test_input)
    
    verify_onnx_model(model_path, test_input_np)


if __name__ == "__main__":
    export_model_to_onnx()
