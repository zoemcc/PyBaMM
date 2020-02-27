import pybamm as pb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

pb.set_logging_level("INFO")

# pick model and parameters
model = pb.lithium_ion.DFN()
model.convert_to_format = "python"
chemistry = pb.parameter_sets.Ecker2015
# chemistry = pb.parameter_sets.Chen2020
# chemistry = pb.parameter_sets.Marquis2019
parameter_values = pb.ParameterValues(chemistry=chemistry)
parameter_values.update({"C-rate": 5})

# set up simulation
sim = pb.Simulation(model, parameter_values=parameter_values)

# solve
sim.solve(solver=pb.CasadiSolver(mode="fast"))

# load in dandeliion data
voltage_data = pd.read_csv("ddliion/voltage.dat", sep="\t")
# t_eval = np.array(voltage_data["t(s)"])

# plot
V = sim.solution["Terminal voltage [V]"]
dc = sim.solution["Discharge capacity [A.h]"]
plt.plot(dc(sim.solution.t), V(sim.solution.t), label="PyBaMM")
plt.plot(voltage_data["Capacity(Ah)"], voltage_data["Voltage(V)"], label="Dandeliion")
plt.xlabel("Dicharge capacity [A.h]")
plt.ylabel("Voltage [V]")
plt.legend()
plt.show()

plt.plot(sim.solution["Time [s]"](sim.solution.t), V(sim.solution.t))
plt.xlabel("Time [s]")
plt.ylabel("Voltage [V]")
plt.legend()
plt.show()
