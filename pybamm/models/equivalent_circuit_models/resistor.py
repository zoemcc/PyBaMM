#
# Resistor model
#
import pybamm
from .base_equivalent_circuit_model import BaseModel


class Resistor(BaseModel):
    """Resistor model of a lithium-ion battery. Here, each through-cell
     electrochemical component of the cell is simple treated as a resistor.

    Parameters
    ----------
    options : dict, optional
        A dictionary of options to be passed to the model.
    name : str, optional
        The name of the model.
    build :  bool, optional
        Whether to build the model on instantiation. Default is True. Setting this
        option to False allows users to change any number of the submodels before
        building the complete model (submodels cannot be changed after the model is
        built).
    """

    def __init__(self, options=None, name="Resistor Model", build=True):
        super().__init__(options, name)

        self.set_external_circuit_submodel()
        self.set_circuit_elements()
        self.set_current_collector_submodel()

        if build:
            self.build_model()

    def set_circuit_elements(self):
        self.submodels["SoC"] = pybamm.circuit_elements.SoC(self.param)
        self.submodels["OCV"] = pybamm.circuit_elements.OCV(self.param)
        self.submodels["R_0"] = pybamm.circuit_elements.LinearResistor(self.param)
