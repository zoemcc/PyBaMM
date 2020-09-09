import pybamm
import numpy as np

pybamm.set_logging_level("DEBUG")
pybamm.settings.debug_mode = True

options = {"dimensionality": 1, "current collector": "potential pair"}
model = pybamm.equivalent_circuit_models.Resistor(options=options)

chemistry = pybamm.parameter_sets.ec_test
parameter_values = pybamm.ParameterValues(chemistry=chemistry)

parameter_values["Initial SoC"] = 0.8
parameter_values["R_0 [Ohm]"] = 0.1
parameter_values["Current function [A]"] = 0.001

var = pybamm.standard_spatial_vars
var_pts = {var.y: 50, var.z: 50}

solver = pybamm.CasadiSolver(mode="safe")
sim = pybamm.Simulation(model, parameter_values=parameter_values, solver=solver)

t_eval = np.linspace(0, 3600, 500)
sim.solve(t_eval)

sim.plot(
    [
        "Terminal voltage [V]",
        "Av SoC",
        "Current collector current density [A.m-2]",
        "Negative current collector potential [V]",
        "Positive current collector potential [V]",
    ]
)
