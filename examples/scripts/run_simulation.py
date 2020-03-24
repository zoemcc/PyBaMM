import pybamm


pybamm.set_logging_level("INFO")

options = {
    "current collector": "potential pair",
    "dimensionality": 2,
    "thermal": "x-lumped",
}
model = pybamm.lithium_ion.DFN(options)

chemistry = pybamm.parameter_sets.NCA_Kim2011
parameter_values = pybamm.ParameterValues(chemistry=chemistry)
parameter_values.update({"Positive tab centre z-coordinate [m]": 0})

var = pybamm.standard_spatial_vars
var_pts = {
    var.x_n: 5,
    var.x_s: 5,
    var.x_p: 5,
    var.r_n: 5,
    var.r_p: 5,
    var.y: 5,
    var.z: 5,
}

sim = pybamm.Simulation(model, parameter_values=parameter_values, var_pts=var_pts)
sim.solve()
sim.plot(
    [
        "X-averaged negative particle surface concentration [mol.m-3]",
        "X-averaged positive particle surface concentration [mol.m-3]",
        "Negative current collector potential [V]",
        "Positive current collector potential [V]",
        "Terminal voltage [V]",
    ]
)
