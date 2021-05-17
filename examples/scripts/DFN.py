import pybamm
import numpy as np
import matplotlib.pyplot as plt

model_dfn = pybamm.lithium_ion.DFN()
model_spm = pybamm.lithium_ion.SPM()

parameter_values = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Chen2020)

# Current frequency
f = 20000  # Discretization effects
# f = 200  # Smooth solution
def my_current(t):
    return pybamm.sin(2 * np.pi * t * f)


parameter_values["Current function [A]"] = my_current
sim_dfn = pybamm.Simulation(model_dfn, parameter_values=parameter_values)
sim_spm = pybamm.Simulation(model_spm, parameter_values=parameter_values)

# 5 cycles
N = 2000  # Number of points in simulation output grid
t_eval = np.linspace(0, 1 / f * 5, N)

sim_dfn.solve(t_eval=t_eval)
sim_spm.solve(t_eval=t_eval)

pybamm.dynamic_plot([sim_dfn, sim_spm], ["Current [A]", "Terminal voltage [V]"])

plt.show()
