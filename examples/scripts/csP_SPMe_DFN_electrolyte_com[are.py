import pybamm
import numpy as np
import matplotlib.pyplot as plt

pybamm.set_logging_level("INFO")
# pybamm.settings.debug_mode = True

C_rate = 5

parameter_values = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Ecker2015)

cSP = pybamm.lithium_ion.BasicCSP()
solver = pybamm.CasadiSolver(mode="fast")
sim_cSP = pybamm.Simulation(
    cSP, parameter_values=parameter_values, C_rate=C_rate, solver=solver
)
sim_cSP.solve()

spme = pybamm.lithium_ion.SPMe()
solver = pybamm.CasadiSolver(mode="fast")
sim_spme = pybamm.Simulation(
    spme, parameter_values=parameter_values, C_rate=C_rate, solver=solver
)
sim_spme.solve()