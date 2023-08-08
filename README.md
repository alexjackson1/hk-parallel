# Parallel Training of Neural Networks with JAX

This repository trains a simple neural network multiple times (with different seed values) on the same device (using JAX).
It is largely a reimplementation of [Parallel Training JAX](https://willwhitney.com/parallel-training-jax.html) by Will Whitney, except this notebook uses `haiku` and `optax` instead of `flax`.

## Installation

The following packages are required to run the notebook:

- JAX (`jax`)
- Haiku (`dm-haiku`)
- Optax (`optax`)

All of the dependencies can be installed using `pip` from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```
