import pybamm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

pybamm.set_logging_level("INFO")
# pybamm.settings.debug_mode = True

C_rate = 5

# matplotlib.rc_file("examples/scripts/_matplotlibrc", use_default_template=True)
parameter_values = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Ecker2015)

var = pybamm.standard_spatial_vars
var_pts = {
    var.x_n: 70,
    var.x_s: 50,
    var.x_p: 70,
    var.r_n: 15,
    var.r_p: 15,
}
var_pts = None  # use defaults

# cSP
cSP = pybamm.lithium_ion.BasicCSP()
solver = pybamm.CasadiSolver(mode="fast")
sim_cSP = pybamm.Simulation(
    cSP,
    parameter_values=parameter_values,
    C_rate=C_rate,
    solver=solver,
    var_pts=var_pts,
)
sim_cSP.solve()

# SPMe LD
spme_LD = pybamm.lithium_ion.BasicSPMe(linear_diffusion=True, use_log=False)
solver = pybamm.CasadiSolver(mode="fast")
sim_spme_LD = pybamm.Simulation(
    spme_LD,
    parameter_values=parameter_values,
    C_rate=C_rate,
    solver=solver,
    var_pts=var_pts,
)
sim_spme_LD.solve()

# SPMe ND
spme_ND = pybamm.lithium_ion.BasicSPMe(linear_diffusion=False, use_log=False)
solver = pybamm.CasadiSolver(mode="fast")
sim_spme_ND = pybamm.Simulation(
    spme_ND,
    parameter_values=parameter_values,
    C_rate=C_rate,
    solver=solver,
    var_pts=var_pts,
)
sim_spme_ND.solve()

# SPMe ND
spme_NDL = pybamm.lithium_ion.BasicSPMe(linear_diffusion=False, use_log=True)
solver = pybamm.CasadiSolver(mode="fast")
sim_spme_NDL = pybamm.Simulation(
    spme_NDL,
    parameter_values=parameter_values,
    C_rate=C_rate,
    solver=solver,
    var_pts=var_pts,
)
sim_spme_NDL.solve()

# DFN
dfn = pybamm.lithium_ion.BasicDFN()
solver = pybamm.CasadiSolver(mode="fast")
sim_dfn = pybamm.Simulation(
    dfn,
    parameter_values=parameter_values,
    C_rate=C_rate,
    solver=solver,
    var_pts=var_pts,
)
sim_dfn.solve()


# Get variables for plotting

t = 0.8 * sim_dfn.solution.t[-1]  # choose point half way through discharge
x = np.linspace(0, 1, 100)
L = parameter_values.process_symbol(pybamm.standard_parameters_lithium_ion.L).evaluate()
x_dim = x * L * 1e6

time = np.round(sim_dfn.solution["Time [s]"](t))

ce_cSP = sim_cSP.solution["Electrolyte concentration [mol.m-3]"](t, x)
ce_SPMe_LD = sim_spme_LD.solution["Electrolyte concentration [mol.m-3]"](t, x)
ce_SPMe_ND = sim_spme_ND.solution["Electrolyte concentration [mol.m-3]"](t, x)
ce_SPMe_NDL = sim_spme_NDL.solution["Electrolyte concentration [mol.m-3]"](t, x)
ce_dfn = sim_dfn.solution["Electrolyte concentration [mol.m-3]"](t, x)

phie_cSP = sim_cSP.solution["Electrolyte potential [V]"](t, x)
phie_SPMe_LD = sim_spme_LD.solution["Electrolyte potential [V]"](t, x)
phie_SPMe_ND = sim_spme_ND.solution["Electrolyte potential [V]"](t, x)
phie_SPMe_NDL = sim_spme_NDL.solution["Electrolyte potential [V]"](t, x)
phie_dfn = sim_dfn.solution["Electrolyte potential [V]"](t, x)


# Set up plotting
fig, ax = plt.subplots(2, 2, figsize=(11, 6))
fig.subplots_adjust(
    left=0.12, bottom=0.1, right=0.77, top=0.92, wspace=0.48, hspace=0.5
)
linestyles = {
    "SPM": "-",
    "DFN": "-",
    "cSP": "--",
    "SPMe (LD)": "--",
    "SPMe (ND)": ":",
    "SPMe (ND+L)": ":",
}

colors = {
    "SPM": "blue",
    "DFN": "black",
    "cSP": "red",
    "SPMe (LD)": "green",
    "SPMe (ND)": "purple",
    "SPMe (ND+L)": (0.99, 0.43, 0.1),
}

# create plots
ax[0, 0].plot(
    x_dim, ce_dfn, label="DFN", linestyle=linestyles["DFN"], color=colors["DFN"]
)
ax[0, 0].plot(
    x_dim, ce_cSP, label="cSP", linestyle=linestyles["cSP"], color=colors["cSP"]
)

if C_rate == 5:
    ax[0, 0].plot(
        x_dim,
        ce_SPMe_LD,
        label="SPMe (LD)",
        linestyle=linestyles["SPMe (LD)"],
        color=colors["SPMe (LD)"],
    )

ax[0, 0].plot(
    x_dim,
    ce_SPMe_ND,
    label="SPMe (ND)",
    linestyle=linestyles["SPMe (ND)"],
    color=colors["SPMe (ND)"],
)
ax[0, 0].plot(
    x_dim,
    ce_SPMe_NDL,
    label="SPMe (ND+L)",
    linestyle=linestyles["SPMe (ND+L)"],
    color=colors["SPMe (ND+L)"],
)

ax[0, 0].set_xlabel("x [$\mu$m]")
ax[0, 0].set_ylabel("Electrolyte concentration [mol.m-3]")


ax[0, 1].plot(
    x_dim, phie_dfn, label="DFN", linestyle=linestyles["DFN"], color=colors["DFN"]
)
ax[0, 1].plot(
    x_dim, phie_cSP, label="cSP", linestyle=linestyles["cSP"], color=colors["cSP"]
)

if C_rate == 5:
    ax[0, 1].plot(
        x_dim,
        phie_SPMe_LD,
        label="SPMe (LD)",
        linestyle=linestyles["SPMe (LD)"],
        color=colors["SPMe (LD)"],
    )
ax[0, 1].plot(
    x_dim,
    phie_SPMe_ND,
    label="SPMe (ND)",
    linestyle=linestyles["SPMe (ND)"],
    color=colors["SPMe (ND)"],
)
ax[0, 1].plot(
    x_dim,
    phie_SPMe_NDL,
    label="SPMe (ND+L)",
    linestyle=linestyles["SPMe (ND+L)"],
    color=colors["SPMe (ND+L)"],
)

ax[0, 1].set_xlabel("x [$\mu$m]")
ax[0, 1].set_ylabel("Electrolyte potential [V]")


def err(var, var_true):
    return np.abs(var - var_true)


ax[1, 0].semilogy(
    x_dim,
    err(ce_cSP, ce_dfn),
    label="cSP",
    linestyle=linestyles["cSP"],
    color=colors["cSP"],
)

if C_rate == 5:
    ax[1, 0].semilogy(
        x_dim,
        err(ce_SPMe_LD, ce_dfn),
        label="SPMe (LD)",
        linestyle=linestyles["SPMe (LD)"],
        color=colors["SPMe (LD)"],
    )
ax[1, 0].semilogy(
    x_dim,
    err(ce_SPMe_ND, ce_dfn),
    label="SPMe (ND)",
    linestyle=linestyles["SPMe (ND)"],
    color=colors["SPMe (ND)"],
)
ax[1, 0].semilogy(
    x_dim,
    err(ce_SPMe_NDL, ce_dfn),
    label="SPMe (ND+L)",
    linestyle=linestyles["SPMe (ND+L)"],
    color=colors["SPMe (ND+L)"],
)

ax[1, 0].set_xlabel("x [$\mu$m]")
ax[1, 0].set_ylabel("Absolute error [mol.m-3]")

ax[1, 1].semilogy(
    x_dim,
    err(phie_cSP, phie_dfn),
    label="cSP vs DFN",
    linestyle=linestyles["cSP"],
    color=colors["cSP"],
)

if C_rate == 5:
    ax[1, 1].semilogy(
        x_dim,
        err(phie_SPMe_LD, phie_dfn),
        label="SPMe (LD) vs DFN",
        linestyle=linestyles["SPMe (LD)"],
        color=colors["SPMe (LD)"],
    )

ax[1, 1].semilogy(
    x_dim,
    err(phie_SPMe_ND, phie_dfn),
    label="SPMe (ND) vs DFN",
    linestyle=linestyles["SPMe (ND)"],
    color=colors["SPMe (ND)"],
)
ax[1, 1].semilogy(
    x_dim,
    err(phie_SPMe_NDL, phie_dfn),
    label="SPMe (ND+L) vs DFN",
    linestyle=linestyles["SPMe (ND+L)"],
    color=colors["SPMe (ND+L)"],
)

ax[1, 1].set_xlabel("x [$\mu$m]")
ax[1, 1].set_ylabel("Absolute error [V]")

# ax[0, 0].legend(loc="upper right")
ax[0, 1].legend(loc="upper left", bbox_to_anchor=(1, 1))
ax[1, 1].legend(loc="upper left", bbox_to_anchor=(1, 1))


# gridlines
ax[0, 0].grid(which="major")
ax[0, 1].grid(which="major")
ax[1, 0].grid(which="major")
ax[1, 1].grid(which="major")

# limits
L_micro = L * 1e6
ax[0, 0].set_xlim([0, L_micro])
ax[0, 1].set_xlim([0, L_micro])
ax[1, 0].set_xlim([0, L_micro])
ax[1, 1].set_xlim([0, L_micro])

ax[0, 0].set_ylim([200, 1800])
ax[0, 1].set_ylim([-0.34, -0.22])
ax[1, 0].set_ylim([1e-2, 1e3])
ax[1, 1].set_ylim([1e-5, 1e-1])


fig.suptitle(str(C_rate) + "C discharge (at " + str(time) + " seconds)")
plt.show()
