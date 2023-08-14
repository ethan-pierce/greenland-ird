"""Quick Python script with micro-benchmarks for JAX."""

import jax
import numpy as np
import lineax as lx

from jax import config

config.update("jax_enable_x64", True)

import time


def main():
    """Run a set of basic performance tests."""
    size = 1000000

    matrix_key, vector_key = jax.random.split(jax.random.PRNGKey(135))
    matrix = jax.random.normal(matrix_key, (size,))
    vector = jax.random.normal(vector_key, (size,))
    operator = lx.DiagonalLinearOperator(matrix)

    starts = []
    ends = []

    for i in range(5):
        starts.append(time.time())
        lx.linear_solve(operator, vector, solver=lx.Diagonal())
        ends.append(time.time())

    print(np.mean(np.array(ends) - np.array(starts)))

if __name__ == "__main__":
    main()
