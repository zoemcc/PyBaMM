#
# Standard parameters for equivalent circuit models
#
import pybamm
import numpy as np


class EquivalentCircuitParameters:
    """
    Standard parameters for equivalent circuit models

    Layout:
        1. Dimensional Parameters
        2. Dimensional Functions
        3. Scalings
        4. Dimensionless Parameters
        5. Dimensionless Functions
        6. Input Current
    """

    def __init__(self, options=None):
        self.options = options

        # Get geometric, electrical and thermal parameters
        self.geo = pybamm.GeometricParameters()
        self.elec = pybamm.ElectricalParameters()
        self.therm = pybamm.ThermalParameters()

        # Set parameters and scales
        self._set_dimensional_parameters()
        self._set_scales()
        self._set_dimensionless_parameters()

        # Set input current
        self._set_input_current()

    def _set_dimensional_parameters(self):
        "Defines the dimensional parameters"

        # Physical constants
        self.R = pybamm.constants.R
        self.F = pybamm.constants.F
        self.T_ref = self.therm.T_ref

        # Macroscale geometry
        self.L_cn = self.geo.L_cn
        self.L_n = self.geo.L_n
        self.L_s = self.geo.L_s
        self.L_p = self.geo.L_p
        self.L_cp = self.geo.L_cp
        self.L_x = self.geo.L_x
        self.L_y = self.geo.L_y
        self.L_z = self.geo.L_z
        self.L = self.geo.L
        self.A_cc = self.geo.A_cc
        self.A_cooling = self.geo.A_cooling
        self.V_cell = self.geo.V_cell

        # Tab geometry
        self.L_tab_n = self.geo.L_tab_n
        self.Centre_y_tab_n = self.geo.Centre_y_tab_n
        self.Centre_z_tab_n = self.geo.Centre_z_tab_n
        self.L_tab_p = self.geo.L_tab_p
        self.Centre_y_tab_p = self.geo.Centre_y_tab_p
        self.Centre_z_tab_p = self.geo.Centre_z_tab_p
        self.A_tab_n = self.geo.A_tab_n
        self.A_tab_p = self.geo.A_tab_p

        # Electrical
        self.I_typ = self.elec.I_typ
        self.Q = self.elec.Q
        self.C_rate = self.elec.C_rate
        self.n_electrodes_parallel = self.elec.n_electrodes_parallel
        self.n_cells = self.elec.n_cells
        self.i_typ = self.elec.i_typ
        self.voltage_low_cut_dimensional = self.elec.voltage_low_cut_dimensional
        self.voltage_high_cut_dimensional = self.elec.voltage_high_cut_dimensional

        # Current Collector
        self.sigma_cn_dimensional = pybamm.Parameter(
            "Negative current collector conductivity [S.m-1]"
        )
        self.sigma_cp_dimensional = pybamm.Parameter(
            "Positive current collector conductivity [S.m-1]"
        )

        # Circuit elements
        self.C_nom_dim = pybamm.Parameter("Nominal capacity [A.h]")
        self.R_0 = pybamm.Parameter("R_0 [Ohm]")

        # Initial conditions
        self.SoC_init = pybamm.Parameter("Initial SoC")

    def OCV_dimensional(self, SoC, T):
        "Dimensional open-circuit voltage"
        inputs = {"SoC": SoC}
        u_ref = pybamm.FunctionParameter("Open-circuit voltage [V]", inputs)
        return u_ref + (T - self.T_ref) * self.dOCVdT_dimensional(SoC)

    def dOCVdT_dimensional(self, SoC):
        """
        Dimensional entropic change of the open-circuit voltage [V.K-1]
        """
        inputs = {
            "SoC": SoC,
        }
        return pybamm.FunctionParameter(
            "Open-circuit voltage entropic change [V.K-1]", inputs
        )

    def _set_scales(self):
        "Define the scales used in the non-dimensionalisation scheme"

        # Electrical
        self.potential_scale = self.R * self.T_ref / self.F
        self.current_scale = self.i_typ

        # Reference OCV based on initial SoC
        self.OCV_ref = self.OCV_dimensional(self.SoC_init, self.T_ref)

        # Thermal
        self.Delta_T = self.therm.Delta_T

        # Discharge timescale
        self.tau_discharge = pybamm.Scalar(3600)  # 1 hour

        # Thermal diffusion timescale
        self.tau_th_yz = self.therm.tau_th_yz

        # Choose discharge timescale
        self.timescale = self.tau_discharge

    def _set_dimensionless_parameters(self):
        "Defines the dimensionless parameters"

        # Timescale ratios
        self.C_th = self.tau_th_yz / self.tau_discharge

        # Macroscale Geometry
        self.l_cn = self.geo.l_cn
        self.l_n = self.geo.l_n
        self.l_s = self.geo.l_s
        self.l_p = self.geo.l_p
        self.l_cp = self.geo.l_cp
        self.l_x = self.geo.l_x
        self.l_y = self.geo.l_y
        self.l_z = self.geo.l_z
        self.a_cc = self.geo.a_cc
        self.a_cooling = self.geo.a_cooling
        self.v_cell = self.geo.v_cell
        self.l = self.geo.l
        self.delta = self.geo.delta

        # Tab geometry
        self.l_tab_n = self.geo.l_tab_n
        self.centre_y_tab_n = self.geo.centre_y_tab_n
        self.centre_z_tab_n = self.geo.centre_z_tab_n
        self.l_tab_p = self.geo.l_tab_p
        self.centre_y_tab_p = self.geo.centre_y_tab_p
        self.centre_z_tab_p = self.geo.centre_z_tab_p

        # Electrical
        self.C_nom = (self.C_nom_dim / self.L_y / self.L_z) / (
            self.i_typ * self.tau_discharge
        )
        self.R_0 = self.R_0 * self.i_typ
        self.eta = pybamm.Parameter("Coulombic efficiency")
        self.voltage_low_cut = (
            self.voltage_low_cut_dimensional - (self.OCV_ref)
        ) / self.potential_scale
        self.voltage_high_cut = (
            self.voltage_high_cut_dimensional - (self.OCV_ref)
        ) / self.potential_scale

        # Current Collector
        self.sigma_cn = (
            self.sigma_cn_dimensional * self.potential_scale / self.i_typ / self.L_x
        )
        self.sigma_cp = (
            self.sigma_cp_dimensional * self.potential_scale / self.i_typ / self.L_x
        )
        self.sigma_cn_prime = self.sigma_cn * self.delta ** 2
        self.sigma_cp_prime = self.sigma_cp * self.delta ** 2
        self.sigma_cn_dbl_prime = self.sigma_cn_prime * self.delta
        self.sigma_cp_dbl_prime = self.sigma_cp_prime * self.delta

        # Thermal
        self.rho_cn = self.therm.rho_cn
        self.rho_n = self.therm.rho_n
        self.rho_s = self.therm.rho_s
        self.rho_p = self.therm.rho_p
        self.rho_cp = self.therm.rho_cp
        self.rho_k = self.therm.rho_k
        self.rho = (
            self.rho_cn * self.l_cn
            + self.rho_n * self.l_n
            + self.rho_s * self.l_s
            + self.rho_p * self.l_p
            + self.rho_cp * self.l_cp
        ) / self.l  # effective volumetric heat capacity

        self.lambda_cn = self.therm.lambda_cn
        self.lambda_n = self.therm.lambda_n
        self.lambda_s = self.therm.lambda_s
        self.lambda_p = self.therm.lambda_p
        self.lambda_cp = self.therm.lambda_cp
        self.lambda_k = self.therm.lambda_k

        self.Theta = self.therm.Theta

        self.h_edge = self.therm.h_edge
        self.h_tab_n = self.therm.h_tab_n
        self.h_tab_p = self.therm.h_tab_p
        self.h_cn = self.therm.h_cn
        self.h_cp = self.therm.h_cp
        self.h_total = self.therm.h_total

        self.B = (
            self.i_typ
            * self.R
            * self.T_ref
            * self.tau_th_yz
            / (self.therm.rho_eff_dim * self.F * self.Delta_T * self.L_x)
        )

        self.T_amb_dim = self.therm.T_amb_dim
        self.T_amb = self.therm.T_amb

    def OCV(self, SoC, T):
        "Dimensionless open-circuit voltage"
        T_dim = self.Delta_T * T + self.T_ref
        return (self.OCV_dimensional(SoC, T_dim) - self.OCV_ref) / self.potential_scale

    def dOCVdT(self, SoC):
        "Dimensionless entropic change in open-circuit voltage"
        return self.dOCVdT_dimensional(SoC) * self.Delta_T / self.potential_scale

    def _set_input_current(self):
        "Set the input current"

        self.dimensional_current_with_time = pybamm.FunctionParameter(
            "Current function [A]", {"Time [s]": pybamm.t * self.timescale}
        )
        self.dimensional_current_density_with_time = (
            self.dimensional_current_with_time
            / (self.n_electrodes_parallel * self.geo.A_cc)
        )
        self.current_with_time = (
            self.dimensional_current_with_time
            / self.I_typ
            * pybamm.Function(np.sign, self.I_typ)
        )
