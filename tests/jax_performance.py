"""Quick Python script with micro-benchmarks for JAX."""

import jax
import lineax as lx

from jax import config
config.update("jax_enable_x64", True)

import time

def main():
    """Run a set of basic performance tests."""
    size = 20000

    matrix_key, vector_key = jax.random.split(jax.random.PRNGKey(135))
    matrix = jax.random.normal(matrix_key, (size, size))
    vector = jax.random.normal(vector_key, (size,))
    operator = lx.MatrixLinearOperator(matrix)

    start = time.time()
    lx.linear_solve(operator, vector, solver=lx.AutoLinearSolver(well_posed = None))
    end = time.time()
    print(end - start)

if __name__ == "__main__":
    main()