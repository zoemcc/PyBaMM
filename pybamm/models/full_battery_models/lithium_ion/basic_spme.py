#
# Basic Single Particle Model with electrolyte (SPMe)
#
import pybamm
from .base_lithium_ion_model import BaseModel


class BasicSPMe(BaseModel):
    """Single Particle Model with electrolyte (SPMe) model of a lithium-ion
    battery, from [2]_.

    This class differs from the :class:`pybamm.lithium_ion.SPMe` model class in that it
    shows the whole model in a single class. This comes at the cost of flexibility in
    combining different physical effects, and in general the main SPMe class should be
    used instead.

    Parameters
    ----------
    name : str, optional
        The name of the model.
    linear_diffusion : str, optional
        Whether or not to use linear diffusion in the electrolyte concentration equation

    References
    ----------
    .. [2] SG Marquis, V Sulzer, R Timms, CP Please and SJ Chapman. “An asymptotic
           derivation of a single particle model with electrolyte”. In: arXiv preprint
           arXiv:1905.12553 (2019).


    **Extends:** :class:`pybamm.lithium_ion.BaseModel`
    """

    def __init__(
        self,
        name="Single Particle Model with electrolyte",
        linear_diffusion=True,
        use_log=False,
    ):
        super().__init__({}, name)
        # `param` is a class containing all the relevant parameters and functions for
        # this model. These are purely symbolic at this stage, and will be set by the
        # `ParameterValues` class when the model is processed.
        param = self.param

        ######################
        # Variables
        ######################
        # Variables that depend on time only are created without a domain
        Q = pybamm.Variable("Discharge capacity [A.h]")
        # Variables that vary spatially are created with a domain
        c_s_n = pybamm.Variable(
            "X-averaged negative particle concentration", domain="negative particle"
        )
        c_s_p = pybamm.Variable(
            "X-averaged positive particle concentration", domain="positive particle"
        )
        c_e_n = pybamm.Variable(
            "Negative electrolyte concentration", domain="negative electrode"
        )
        c_e_s = pybamm.Variable(
            "Separator electrolyte concentration", domain="separator"
        )
        c_e_p = pybamm.Variable(
            "Positive electrolyte concentration", domain="positive electrode"
        )
        # Concatenations combine several variables into a single variable, to simplify
        # implementing equations that hold over several domains
        c_e = pybamm.Concatenation(c_e_n, c_e_s, c_e_p)

        # Constant temperature
        T = param.T_init

        ######################
        # Other set-up
        ######################

        # Current density
        i_cell = param.current_with_time
        j_n = i_cell / param.l_n
        j_p = -i_cell / param.l_p

        # Porosity
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
        tor_n = eps_n ** param.b_e_n
        tor_s = eps_s ** param.b_e_s
        tor_p = eps_p ** param.b_e_p
        tor = pybamm.Concatenation(tor_n, tor_s, tor_p)

        ######################
        # State of Charge
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
        c_s_surf_n = pybamm.surf(c_s_n)
        c_s_surf_p = pybamm.surf(c_s_p)
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
        # c_n_init and c_p_init are functions, but for the SPMe we evaluate them at x=0
        # and x=1 since there is no x-dependence in the particles
        self.initial_conditions[c_s_n] = param.c_n_init(0)
        self.initial_conditions[c_s_p] = param.c_p_init(1)

        ######################
        # Electrolyte concentration
        ######################
        if linear_diffusion is True:
            # For the linear SPMe we evaluate the diffusivity at the typical value
            N_e = -tor * param.D_e(1, T) * pybamm.grad(c_e)
        elif linear_diffusion is False:
            # Otherwise evaluate diffusivity using c_e
            N_e = -tor * param.D_e(c_e, T) * pybamm.grad(c_e)
        else:
            raise pybamm.OptionError("linear_diffusion must be either True or False")

        # We create a concatenation for the reaction current
        j = pybamm.Concatenation(
            pybamm.PrimaryBroadcast(j_n, "negative electrode"),
            pybamm.PrimaryBroadcast(0, "separator"),
            pybamm.PrimaryBroadcast(j_p, "positive electrode"),
        )
        self.rhs[c_e] = (1 / eps) * (
            -pybamm.div(N_e) / param.C_e + (1 - param.t_plus(c_e)) * j / param.gamma_e
        )
        self.boundary_conditions[c_e] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (pybamm.Scalar(0), "Neumann"),
        }
        self.initial_conditions[c_e] = param.c_e_init

        ######################
        # (Some) variables
        ######################

        if use_log is False:
            fun = linear_function
        elif use_log is True:
            fun = log_function
        else:
            raise pybamm.OptionError("use_log must be either True or False")

        U_n = param.U_n(c_s_surf_n, T)
        U_p = param.U_p(c_s_surf_p, T)
        ocv = U_p - U_n

        j0_n = pybamm.x_average(
            param.m_n(T)
            / param.C_r_n
            * c_s_surf_n ** (1 / 2)
            * (1 - c_s_surf_n) ** (1 / 2)
            * (c_e_n) ** (1 / 2)
        )
        j0_p = pybamm.x_average(
            param.gamma_p
            * param.m_p(T)
            / param.C_r_p
            * c_s_surf_p ** (1 / 2)
            * (1 - c_s_surf_p) ** (1 / 2)
            * (c_e_p) ** (1 / 2)
        )
        eta_n = (2 * (1 + self.param.Theta * T) / param.ne_n) * pybamm.arcsinh(
            j_n / (2 * j0_n)
        )
        eta_p = (2 * (1 + self.param.Theta * T) / param.ne_p) * pybamm.arcsinh(
            j_p / (2 * j0_p)
        )
        eta_r = eta_p - eta_n
        eta_c = (
            2
            * (1 - param.t_plus(pybamm.x_average(c_e)))
            * (pybamm.x_average(fun(c_e_p, c_e)) - pybamm.x_average(fun(c_e_n, c_e)))
        )
        delta_phi_e_av = -(
            param.C_e * i_cell / (param.gamma_e * param.kappa_e(1, T))
        ) * (
            pybamm.x_average(param.l_n / (3 * eps_n ** param.b_e_n))
            + pybamm.x_average(param.l_s / (eps_s ** param.b_e_s))
            + pybamm.x_average(param.l_p / (3 * eps_p ** param.b_e_p))
        )

        delta_phi_s_av = -(i_cell / 3) * (
            param.l_p / param.sigma_p + param.l_n / param.sigma_n
        )
        V = ocv + eta_r + eta_c + delta_phi_e_av + delta_phi_s_av

        # Domains at cell centres
        x_n = pybamm.SpatialVariable(
            "x_n", domain=["negative electrode"], coord_sys="cartesian",
        )
        x_s = pybamm.SpatialVariable(
            "x_s", domain=["separator"], coord_sys="cartesian",
        )
        x_p = pybamm.SpatialVariable(
            "x_p", domain=["positive electrode"], coord_sys="cartesian",
        )

        eps_n_av = pybamm.x_average(eps_n)
        eps_s_av = pybamm.x_average(eps_s)
        eps_p_av = pybamm.x_average(eps_p)


        phi_s_n_av = -(i_cell / 3) * param.l_n / param.sigma_n

        c_e_av = pybamm.x_average(c_e)

        # Electrolyte potentials
        phi_e_const = (
            phi_s_n_av
            - U_n
            - eta_n
            - 2
            * (1 - param.t_plus(c_e_av))
            * (1 + param.Theta * T)
            * pybamm.x_average(fun(c_e_n, c_e))
            - (
                (
                    i_cell
                    * param.C_e
                    * param.l_n
                    / (param.gamma_e * param.kappa_e(c_e_av, T))
                )
                * (1 / (3 * eps_n_av ** param.b_e_n) - 1 / eps_s_av ** param.b_e_s)
            )
        )

        phi_e_n = (
            phi_e_const
            + 2 * (1 - param.t_plus(c_e_av)) * (1 + param.Theta * T) * fun(c_e_n, c_e)
            - i_cell
            * param.C_e
            / (param.gamma_e * param.kappa_e(1, T))
            * (x_n ** 2 - param.l_n ** 2)
            / (2 * param.l_n * eps_n_av ** param.b_e_n)
            - i_cell
            * param.l_n
            * (param.C_e / param.gamma_e)
            / param.kappa_e(c_e_av, T)
            / (eps_s_av ** param.b_e_s)
        )

        phi_e_s = (
            phi_e_const
            + 2 * (1 - param.t_plus(c_e_av)) * (1 + param.Theta * T) * fun(c_e_s, c_e)
            - (
                i_cell
                * param.C_e
                / param.gamma_e
                / (param.kappa_e(c_e_av, T) * eps_s_av ** param.b_e_s)
                * x_s
            )
        )

        phi_e_p = (
            phi_e_const
            + 2 * (1 - param.t_plus(c_e_av)) * (1 + param.Theta * T) * fun(c_e_p, c_e)
            - (
                i_cell
                * (param.C_e / param.gamma_e)
                / (param.kappa_e(c_e_av, T) * eps_p_av ** param.b_e_p)
            )
            * (x_p * (2 - x_p) + param.l_p ** 2 - 1)
            / (2 * param.l_p)
            - i_cell
            * (1 - param.l_p)
            * (param.C_e / param.gamma_e)
            / (param.kappa_e(1, T) * eps_s_av ** param.b_e_s)
        )

        phi_e = pybamm.Concatenation(phi_e_n, phi_e_s, phi_e_p)

        self.variables = {
            "Electrolyte concentration": c_e,
            "Electrolyte concentration [mol.m-3]": c_e * param.c_e_typ,
            "Electrolyte potential [V]": phi_e * param.potential_scale - param.U_n_ref,
            "Negative electrode potential [V]": pybamm.PrimaryBroadcast(
                phi_s_n_av, ["negative electrode"]
            )
            * param.potential_scale,
            "Terminal voltage": V,
            "Terminal voltage [V]": param.U_p_ref
            - param.U_n_ref
            + param.potential_scale * V,
            "Real terminal voltage [V]": param.U_p_ref
            - param.U_n_ref
            + param.potential_scale * V,
            "Time [s]": pybamm.t * param.timescale,
        }

        self.events += [
            pybamm.Event("Minimum voltage", V - param.voltage_low_cut),
            pybamm.Event("Maximum voltage", V - param.voltage_high_cut),
        ]

    @property
    def default_geometry(self):
        return pybamm.Geometry("1D macro", "1D micro")


def linear_function(c_e, c_e_full):
    return c_e - pybamm.x_average(c_e_full)


def log_function(c_e, c_e_full):
    c_e_av = pybamm.x_average(c_e_full)
    return pybamm.log(c_e / c_e_av)
