# Benchmarking

Comparison of solvers (integrators) for different models with varying mesh sizes. In all cases, each subdomain has the same number of grid points.

## Lithium-ion models

### Single Particle Model

 Grid points | Scipy | Scikits ODE | Scikits DAE |
---|---|---|---|
 100 | 0.38 seconds | 0.72 seconds | 1.01 seconds |
 200 | 0.36 seconds | 0.58 seconds | 1.58 seconds |
 400 | 0.43 seconds | 1.42 seconds | 5.25 seconds |

## Lead-acid models

### LOQS model

 Grid points | Scipy | Scikits ODE | Scikits DAE |
---|---|---|---|
 1000 | 0.03 seconds | 0.05 seconds | 0.07 seconds |
 2000 | 0.03 seconds | 0.05 seconds | 0.08 seconds |
 4000 | 0.02 seconds | 0.05 seconds | 0.07 seconds |

