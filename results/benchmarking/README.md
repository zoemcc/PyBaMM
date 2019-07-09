# Benchmarking

Comparison of solvers (time-steppers) for different models with varying mesh sizes

## Lithium-ion models

### Single particle model

  | Scipy | Scikits ODE | Scikits DAE |
---|---|---|---|
 100 | 0.26 seconds | 0.28 seconds | 0.47 seconds |
 200 | 0.31 seconds | 0.55 seconds | 1.49 seconds |
 400 | 0.29 seconds | 1.72 seconds | 5.38 seconds |

## Lead-acid models

### LOQS model

  | Scipy | Scikits ODE | Scikits DAE |
---|---|---|---|
 1000 | 0.03 seconds | 0.04 seconds | 0.07 seconds |
 2000 | 0.03 seconds | 0.05 seconds | 0.1 seconds |
 4000 | 0.03 seconds | 0.06 seconds | 0.09 seconds |

### Newman-Tiedemann model

  | Scikits DAE |
---|---|
 10 | 1.2 seconds |
 20 | 1.61 seconds |
 40 | 14.33 seconds |

