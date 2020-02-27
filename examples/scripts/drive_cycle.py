import pybamm

pybamm.set_logging_level("INFO")
model = pybamm.lithium_ion.DFN()
param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Ecker2015)
param["Current function [A]"] = "[current data]US06"
sim = pybamm.Simulation(model, parameter_values=param, solver=pybamm.IDAKLUSolver())
sim.solve()
sim.plot()
