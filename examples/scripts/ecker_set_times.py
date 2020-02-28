import pybamm as pb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

pb.set_logging_level("INFO")

# set up different solvers
solvers = [
    # pb.ScikitsDaeSolver(),
    pb.IDAKLUSolver(),
    pb.CasadiSolver(mode="safe"),
    pb.CasadiSolver(mode="fast"),
]

# pick model and parameters
model = pb.lithium_ion.DFN()
model.convert_to_format = "python"
chemistry = pb.parameter_sets.Ecker2015
# chemistry = pb.parameter_sets.Chen2020
# chemistry = pb.parameter_sets.Marquis2019
parameter_values = pb.ParameterValues(chemistry=chemistry)
drive_cycle = True

# pick npts per domain
npts = [10, 20, 40]
var = pb.standard_spatial_vars

if drive_cycle:
    parameter_values["Current function [A]"] = "[current data]US06"
    t_eval = None
else:
    t_eval = np.linspace(0, 3780, 63)

# set up data for table
data = np.array([[None] * (len(solvers) + 2)] * len(npts))

for i_N, N in enumerate(npts):
    print("N = {}".format(N))
    var_pts = {var.x_n: N, var.x_s: N, var.x_p: N, var.r_n: N, var.r_p: N}

    # set up simulation
    sim = pb.Simulation(model, parameter_values=parameter_values, var_pts=var_pts)
    sim.build()

    # save npts and number of states
    states = sim._built_model.concatenated_initial_conditions.shape[0]
    data[i_N][0] = N
    data[i_N][1] = states

    # solve for different solvers
    times = [None] * len(solvers)
    for i, solver in enumerate(solvers):
        print("Solver = {}".format(solver.name))
        try:
            sim.solve(solver=solver, t_eval=t_eval)
            times[i] = sim.solution.solve_time
        except:
            pass

    # add solve time data
    data[i_N][2:] = times

# make table
headers = ["N", "States", *[solver.name for solver in solvers]]
print(tabulate(data, headers=headers))
