import pybamm

pybamm.set_logging_level("DEBUG")

model = pybamm.equivalent_circuit_models.Resistor()


chemistry = pybamm.parameter_sets.ec_test
parameter_values = pybamm.ParameterValues(chemistry=chemistry)

sim = pybamm.Simulation(model)

sim.solve([0, 3500])
