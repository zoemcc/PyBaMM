#
# Class for electrolyte diffusion employing stefan-maxwell
#
import pybamm

from .base_electrolyte_diffusion import BaseElectrolyteDiffusion


class Full(BaseElectrolyteDiffusion):
    """Class for conservation of mass in the electrolyte employing the
    Stefan-Maxwell constitutive equations. (Full refers to unreduced by
    asymptotic methods)

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    options : dict, optional
        A dictionary of options to be passed to the model.

    **Extends:** :class:`pybamm.electrolyte_diffusion.BaseElectrolyteDiffusion`
    """

    def __init__(self, param, options=None):
        super().__init__(param, options)

    def get_fundamental_variables(self):
        """
        if self.half_cell:
            eps_c_e_n = None
        else:
            eps_c_e_n = pybamm.standard_variables.eps_c_e_n
        eps_c_e_s = pybamm.standard_variables.eps_c_e_s
        eps_c_e_p = pybamm.standard_variables.eps_c_e_p

        variables = self._get_standard_porosity_times_concentration_variables(
            eps_c_e_n, eps_c_e_s, eps_c_e_p
        )
        """

        if self.half_cell:
            c_e_n = None
        else:
            c_e_n = pybamm.standard_variables.c_e_n
        c_e_s = pybamm.standard_variables.c_e_s
        c_e_p = pybamm.standard_variables.c_e_p

        variables = self._get_standard_concentration_variables(c_e_n, c_e_s, c_e_p)

        return variables

    def get_coupled_variables(self, variables):

        """
        if self.half_cell:
            c_e_n = None
        else:
            eps_n = variables["Negative electrode porosity"]
            eps_c_e_n = variables["Negative electrode porosity times concentration"]
            c_e_n = eps_c_e_n / eps_n

        eps_s = variables["Separator porosity"]
        eps_p = variables["Positive electrode porosity"]
        eps_c_e_s = variables["Separator porosity times concentration"]
        eps_c_e_p = variables["Positive electrode porosity times concentration"]
        c_e_s = eps_c_e_s / eps_s
        c_e_p = eps_c_e_p / eps_p

        variables.update(
            self._get_standard_concentration_variables(c_e_n, c_e_s, c_e_p)
        )
        """

        if self.half_cell:
            eps_c_e_n = None
        else:
            eps_n = variables["Negative electrode porosity"]
            c_e_n = variables["Negative electrolyte concentration"]
            eps_c_e_n = c_e_n * eps_n

        eps_s = variables["Separator porosity"]
        eps_p = variables["Positive electrode porosity"]
        c_e_s = variables["Separator electrolyte concentration"]
        c_e_p = variables["Positive electrolyte concentration"]
        eps_c_e_s = c_e_s * eps_s
        eps_c_e_p = c_e_p * eps_p

        variables.update(
            self._get_standard_porosity_times_concentration_variables(eps_c_e_n, eps_c_e_s, eps_c_e_p)
        )

        eps_c_e = variables["Porosity times concentration"]
        c_e = variables["Electrolyte concentration"]
        tor = variables["Electrolyte transport efficiency"]
        i_e = variables["Electrolyte current density"]
        v_box = variables["Volume-averaged velocity"]
        T = variables["Cell temperature"]

        param = self.param

        N_e_diffusion = -tor * param.D_e(c_e, T) * pybamm.grad(c_e)
        N_e_migration = param.C_e * param.t_plus(c_e, T) * i_e / param.gamma_e
        N_e_convection = param.C_e * c_e * v_box

        N_e = N_e_diffusion + N_e_migration + N_e_convection

        variables.update(self._get_standard_flux_variables(N_e))
        variables.update(self._get_total_concentration_electrolyte(eps_c_e))

        return variables

    def set_rhs(self, variables):

        param = self.param

        eps = variables["Porosity"]
        eps_c_e = variables["Porosity times concentration"]
        c_e = variables["Electrolyte concentration"]
        N_e = variables["Electrolyte flux"]
        div_Vbox = variables["Transverse volume-averaged acceleration"]

        sum_s_j = variables["Sum of electrolyte reaction source terms"]
        sum_s_j.print_name = "a"
        source_terms = sum_s_j / self.param.gamma_e

        self.rhs = {
            c_e: (-pybamm.div(N_e) / param.C_e + source_terms - c_e * div_Vbox) / eps # I think this is fine???
            #eps_c_e: -pybamm.div(N_e) / param.C_e + source_terms - c_e * div_Vbox
        }

    def set_initial_conditions(self, variables):

        #eps_c_e = variables["Porosity times concentration"]
        c_e = variables["Electrolyte concentration"]

        self.initial_conditions = {
            #eps_c_e: self.param.epsilon_init * self.param.c_e_init
            c_e: self.param.c_e_init
        }

    def set_boundary_conditions(self, variables):
        param = self.param
        c_e = variables["Electrolyte concentration"]
        eps_c_e = variables["Porosity times concentration"]

        if self.half_cell:
            # left bc at anode/separator interface
            # assuming v_box = 0 for now
            T = variables["Cell temperature"]
            tor = variables["Electrolyte transport efficiency"]
            i_boundary_cc = variables["Current collector current density"]
            #eps_s = variables["Separator porosity"]
            lbc = (
                pybamm.boundary_value(
                    -(1 - param.t_plus(c_e, T))
                    #/ (tor * param.gamma_e * param.D_e(c_e, T) * eps_s),
                    / (tor * param.gamma_e * param.D_e(c_e, T)),
                    "left",
                )
                * i_boundary_cc
                * param.C_e
            )
        else:
            # left bc at anode/current collector interface
            lbc = pybamm.Scalar(0)

        self.boundary_conditions = {
            #eps_c_e: {"left": (pybamm.Scalar(0), "Neumann"), "right": (pybamm.Scalar(0), "Neumann")},
            c_e: {"left": (lbc, "Neumann"), "right": (pybamm.Scalar(0), "Neumann")},
        }
