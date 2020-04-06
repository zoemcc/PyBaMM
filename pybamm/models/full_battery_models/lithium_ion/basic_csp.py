#
# Basic Single Particle Model with electrolyte (SPMe)
#
import pybamm
from .base_lithium_ion_model import BaseModel


class BasicCSP(BaseModel):
    """Corrected Single Particle Model (SPM) model of a lithium-ion battery, from ??.

    This class differs from the :class:`pybamm.lithium_ion.SPM` model class in that it
    shows the whole model in a single class. This comes at the cost of flexibility in
    combining different physical effects, and in general the main SPM class should be
    used instead.

    Parameters
    ----------
    name : str, optional
        The name of the model.

    References
    ----------


    **Extends:** :class:`pybamm.lithium_ion.BaseModel`
    """

    def __init__(self, name="cSP"):
        super().__init__({}, name)
        # `param` is a class containing all the relevant parameters and functions for
        # this model. These are purely symbolic at this stage, and will be set by the
        # `ParameterValues` class when the model is processed.
        param = self.param
        self.timescale = param.tau_discharge

        ######################
        # Variables
        ######################
        # Variables that depend on time only are created without a domain
        Q = pybamm.Variable("Discharge capacity [A.h]")
        # Variables that vary spatially are created with a domain

        # Particle concentration
        c_s_n = pybamm.Variable(
            "X-averaged negative particle concentration", domain="negative particle"
        )
        c_s_p = pybamm.Variable(
            "X-averaged positive particle concentration", domain="positive particle"
        )

        # Electrolyte concentrations
        c_e_n = pybamm.Variable(
            "Negative electrolyte concentration", domain="negative electrode"
        )
        c_e_s = pybamm.Variable(
            "Separator electrolyte concentration", domain="separator"
        )
        c_e_p = pybamm.Variable(
            "Positive electrolyte concentration", domain="positive electrode"
        )
        c_e = pybamm.Concatenation(c_e_n, c_e_s, c_e_p)

        # Electrolyte potentials
        phi_e_n = pybamm.Variable(
            "Negative electrolyte potential", domain="negative electrode"
        )
        phi_e_s = pybamm.Variable("Separator electrolyte potential", domain="separator")
        phi_e_p = pybamm.Variable(
            "Positive electrolyte potential", domain="positive electrode"
        )
        phi_e = pybamm.Concatenation(phi_e_n, phi_e_s, phi_e_p)

        # Constant temperature
        T = param.T_init

        ######################
        # Coupled variables
        ######################

        # Current density
        i_cell = param.current_with_time
        j_n = i_cell / param.l_n
        j_p = -i_cell / param.l_p
        j = pybamm.Concatenation(
            pybamm.PrimaryBroadcast(j_n, "negative electrode"),
            pybamm.PrimaryBroadcast(0, "separator"),
            pybamm.PrimaryBroadcast(j_p, "positive electrode"),
        )

        # Porosity
        # Primary broadcasts are used to broadcast scalar quantities across a domain
        # into a vector of the right shape, for multiplying with other vectors
        eps_n = pybamm.PrimaryBroadcast(
            pybamm.Parameter("Negative electrode porosity"), "negative electrode"
        )
        eps_s = pybamm.PrimaryBroadcast(
            pybamm.Parameter("Separator porosity"), "separator"
        )
        eps_p = pybamm.PrimaryBroadcast(
            pybamm.Parameter("Positive electrode porosity"), "positive electrode"
        )
        eps = pybamm.Concatenation(eps_n, eps_s, eps_p)

        # Tortuosity
        tor = pybamm.Concatenation(
            eps_n ** param.b_e_n, eps_s ** param.b_e_s, eps_p ** param.b_e_p
        )
        tor_n, tor_s, tor_p = tor.orphans

        # surface concentrations
        c_s_surf_n = pybamm.surf(c_s_n)
        c_s_surf_p = pybamm.surf(c_s_p)

        # OCPs
        U_n = param.U_n(c_s_surf_n, T)
        U_p = param.U_p(c_s_surf_p, T)

        # Interfacial reactions
        j0_n = (
            param.m_n(T)
            / param.C_r_n
            * c_s_surf_n ** (1 / 2)
            * (1 - c_s_surf_n) ** (1 / 2)
            * (c_e_n) ** (1 / 2)
        )

        j0_p = (
            param.gamma_p
            * param.m_p(T)
            / param.C_r_p
            * c_s_surf_p ** (1 / 2)
            * (1 - c_s_surf_p) ** (1 / 2)
            * (c_e_p) ** (1 / 2)
        )

        # Electrolyte current
        i_e_n = (param.kappa_e(c_e_n, T) * tor_n * param.gamma_e / param.C_e) * (
            param.chi(c_e_n) * (1 + param.Theta * T) * pybamm.grad(c_e_n) / c_e_n
            - pybamm.grad(phi_e_n)
        )
        i_e_p = (param.kappa_e(c_e_p, T) * tor_p * param.gamma_e / param.C_e) * (
            param.chi(c_e_p) * (1 + param.Theta * T) * pybamm.grad(c_e_p) / c_e_p
            - pybamm.grad(phi_e_p)
        )

        i_e = (param.kappa_e(c_e, T) * tor * param.gamma_e / param.C_e) * (
            param.chi(c_e) * (1 + param.Theta * T) * pybamm.grad(c_e) / c_e
            - pybamm.grad(phi_e)
        )

        # Electrode currents
        j_s_n = i_cell - i_e_n
        j_s_p = i_cell - i_e_p

        # Electrode potentials
        phi_s_n = -pybamm.IndefiniteIntegral(
            j_s_n, pybamm.standard_spatial_vars.x_n
        ) / (param.sigma_n)

        # (need to do this as can't x_average on edge variables...)
        x_p = pybamm.SpatialVariable(
            "x_p", domain=["positive electrode"], coord_sys="cartesian",
        )
        j_s_p_anal = i_cell - j_p * (x_p - param.l) / (param.l_p)
        # phi_s_p = int_x^L4  but write as int_L3^L4 - int_L3^x
        phi_s_p = (
            pybamm.x_average(j_s_p_anal)
            - pybamm.IndefiniteIntegral(j_s_p, pybamm.standard_spatial_vars.x_p)
        ) / param.sigma_p

        # Negative ion flux (N.B they solve for N- but we normally solve for N+)
        N_e = (
            -tor * param.D_e(c_e, T) * pybamm.grad(c_e)
            - (1 - param.t_plus(c_e)) * i_e * param.C_e / param.gamma_e
        )

        eta_n = pybamm.x_average((2 / param.ne_n) * pybamm.arcsinh(j_n / (2 * j0_n)))
        eta_p = pybamm.x_average((2 / param.ne_p) * pybamm.arcsinh(j_p / (2 * j0_p)))

        V_n = U_n + eta_n + pybamm.x_average(phi_e_n) - pybamm.x_average(phi_s_n)
        V_p = U_p + eta_p + pybamm.x_average(phi_e_p) - pybamm.x_average(phi_s_p)

        V = V_p - V_n

        ######################
        # Governing Equations
        ######################
        I = param.dimensional_current_with_time
        # The `rhs` dictionary contains differential equations, with the key being the
        # variable in the d/dt
        self.rhs[Q] = I * param.timescale / 3600
        # Initial conditions must be provided for the ODEs
        self.initial_conditions[Q] = pybamm.Scalar(0)

        ######################
        # Particles
        ######################

        # The div and grad operators will be converted to the appropriate matrix
        # multiplication at the discretisation stage
        N_s_n = -param.D_n(c_s_n, T) * pybamm.grad(c_s_n)
        N_s_p = -param.D_p(c_s_p, T) * pybamm.grad(c_s_p)
        self.rhs[c_s_n] = -(1 / param.C_n) * pybamm.div(N_s_n)
        self.rhs[c_s_p] = -(1 / param.C_p) * pybamm.div(N_s_p)
        # Surf takes the surface value of a variable, i.e. its boundary value on the
        # right side. This is also accessible via `boundary_value(x, "right")`, with
        # "left" providing the boundary value of the left side
        # Boundary conditions must be provided for equations with spatial derivatives
        self.boundary_conditions[c_s_n] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (
                -param.C_n * j_n / param.a_n / param.D_n(c_s_surf_n, T),
                "Neumann",
            ),
        }
        self.boundary_conditions[c_s_p] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (
                -param.C_p * j_p / param.a_p / param.gamma_p / param.D_p(c_s_surf_p, T),
                "Neumann",
            ),
        }
        # c_n_init and c_p_init are functions, but for the SPM we evaluate them at x=0
        # and x=1 since there is no x-dependence in the particles
        self.initial_conditions[c_s_n] = param.c_n_init(0)
        self.initial_conditions[c_s_p] = param.c_p_init(1)

        # Electrolyte - derived variables

        ######################
        # Electrolyte concentration
        ######################
        self.rhs[c_e] = (1 / eps) * (-pybamm.div(N_e) / param.C_e)
        self.boundary_conditions[c_e] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (pybamm.Scalar(0), "Neumann"),
        }
        self.initial_conditions[c_e] = param.c_e_init

        ######################
        # Electrolyte potential
        ######################
        self.algebraic[phi_e] = pybamm.div(i_e) - j
        self.boundary_conditions[phi_e] = {
            "left": (pybamm.Scalar(0), "Dirichlet"),
            "right": (pybamm.Scalar(0), "Neumann"),
        }
        self.initial_conditions[phi_e] = -self.param.U_n(self.param.c_n_init(0), T)

        ######################

        ######################
        # (Some) variables
        ######################

        # correct phi_e
        phi_e_ref = -eta_n + pybamm.x_average(phi_s_n) - U_n

        phi_e = phi_e - pybamm.x_average(phi_e_n) + phi_e_ref
        phi_e_n = phi_e_n - pybamm.x_average(phi_e_n) + phi_e_ref
        phi_e_s = phi_e_s - pybamm.x_average(phi_e_n) + phi_e_ref
        phi_e_p = phi_e_p - pybamm.x_average(phi_e_n) + phi_e_ref

        c_e_typ = pybamm.standard_parameters_lithium_ion.c_e_typ

        self.variables = {
            "phi e ref": phi_e_ref,
            "Electrolyte concentration": c_e,
            "Electrolyte concentration [mol.m-3]": c_e * c_e_typ,
            "Negative electrode potential": phi_s_n,
            "Negative electrode potential [V]": phi_s_n * param.potential_scale,
            "Positive electrode potential drop": phi_s_p * param.potential_scale
            + param.U_p_ref
            - param.U_n_ref,
            "Negative reaction overpotential": eta_n,
            "Positive reaction overpotential": eta_p,
            "Negative electrolyte potential [V]": phi_e_n * param.potential_scale
            - param.U_n_ref,
            "Positive electrolyte potential [V]": phi_e_p * param.potential_scale
            - param.U_n_ref,
            "Positive electrode potential [V]": (phi_s_p + V) * param.potential_scale
            + param.U_p_ref
            - param.U_n_ref,
            "Electrolyte current density": i_e,
            "Electrolyte potential [V]": phi_e * param.potential_scale - param.U_n_ref,
            "Terminal voltage": V,
            "Terminal voltage [V]": param.U_p_ref
            - param.U_n_ref
            + param.potential_scale * V,
            "Time [s]": pybamm.t * self.timescale,
        }
        self.events += [
            pybamm.Event("Minimum voltage", V - param.voltage_low_cut),
            pybamm.Event("Maximum voltage", V - param.voltage_high_cut),
        ]

    @property
    def default_geometry(self):
        return pybamm.Geometry("1D macro", "1D micro")
