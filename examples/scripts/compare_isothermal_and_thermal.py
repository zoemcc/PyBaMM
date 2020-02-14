import pybamm
import numpy as np

# load models
options_isothermal = {"thermal": "isothermal"}  # , "particle": "fast diffusion"}
options_thermal = {"thermal": "x-lumped"}  # , "particle": "fast diffusion"}
models = [
    pybamm.lithium_ion.SPM(options_isothermal),  # isothermal
    pybamm.lithium_ion.SPM(options_thermal),  # x-lumped thermal
]

# load parameter values and process models and geometry
param = models[0].default_parameter_values
param["Heat transfer coefficient [W.m-2.K-1]"] = 0

for model in models:
    param.process_model(model)

# set mesh
var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 10, var.x_s: 10, var.x_p: 10, var.r_n: 15, var.r_p: 15}


# discretise models
for model in models:

    # model.variables[
    #     "Diffusion coefficient"
    # ] = pybamm.standard_parameters_lithium_ion.D_n(
    #     model.variables["Negative particle concentration"], T_k_xav,
    # )
    # create geometry
    geometry = model.default_geometry
    param.process_geometry(geometry)
    mesh = pybamm.Mesh(geometry, models[-1].default_submesh_types, var_pts)
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)


# solve model
solutions = [None] * len(models)
t_eval = np.linspace(0, 0.3, 100)
for i, model in enumerate(models):
    solutions[i] = model.default_solver.solve(model, t_eval)

# D_n_isothermal = solutions[0]["Diffusion coefficient"]
# D_n_thermal = solutions[1]["Diffusion coefficient"]


# plot
output_variables = [
    "X-averaged negative particle concentration",
    "X-averaged positive particle concentration",
    "Negative particle surface concentration",
    "Positive particle surface concentration",
    "Negative electrode interfacial current density",
    "Positive electrode interfacial current density",
    "Terminal voltage [V]",
    "Current [A]",
    "X-averaged negative particle flux",
    "X-averaged positive particle flux"
]
plot = pybamm.QuickPlot(solutions, output_variables=output_variables)
plot.dynamic_plot()
