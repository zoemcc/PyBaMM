#
# Base equivalent circuit model class
#

import pybamm


class BaseEquivalentCircuitModel(pybamm.BaseModel):
    """
    Base model class for equivalent circuit models with some default settings
    and required variables.

    Attributes
    ----------

    options: dict
        A dictionary of options to be passed to the model. The options that can
        be set are listed below. Note that not all of the options are compatible with
        each other and with all of the models implemented in PyBaMM.

            * "cell geometry" : str, optional
                Sets the geometry of the cell. Can be "pouch" (default) or
                "arbitrary". The arbitrary geometry option solves a 1D electrochemical
                model with prescribed cell volume and cross-sectional area, and
                (if thermal effects are included) solves a lumped thermal model
                with prescribed surface area for cooling.
            * "dimensionality" : int, optional
                Sets the dimension of the current collector problem. Can be 0
                (default), 1 or 2.
            * "current collector" : str, optional
                Sets the current collector model to use. Can be "uniform" (default),
                "potential pair" or "potential pair quite conductive".

    **Extends:** :class:`pybamm.BaseModel`
    """

    def __init__(self, options=None, name="Unnamed equivalent circuit model"):
        super().__init__(name)
        self.options = options
        self.submodels = {}
        self._built = False
        self._built_fundamental_and_external = False

    @property
    def default_parameter_values(self):
        # Default parameter values
        # Lion parameters left as default parameter set for tests
        return pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Marquis2019)

    @property
    def default_geometry(self):
        return pybamm.battery_geometry(
            current_collector_dimension=self.options["dimensionality"]
        )

    @property
    def default_var_pts(self):
        var = pybamm.standard_spatial_vars
        base_var_pts = {
            var.y: 10,
            var.z: 10,
        }
        return base_var_pts

    @property
    def default_submesh_types(self):
        base_submeshes = {}
        if self.options["dimensionality"] == 0:
            base_submeshes["current collector"] = pybamm.MeshGenerator(pybamm.SubMesh0D)
        elif self.options["dimensionality"] == 1:
            base_submeshes["current collector"] = pybamm.MeshGenerator(
                pybamm.Uniform1DSubMesh
            )
        elif self.options["dimensionality"] == 2:
            base_submeshes["current collector"] = pybamm.MeshGenerator(
                pybamm.ScikitUniform2DSubMesh
            )
        return base_submeshes

    @property
    def default_spatial_methods(self):
        base_spatial_methods = {}
        if self.options["dimensionality"] == 0:
            # 0D submesh - use base spatial method
            base_spatial_methods[
                "current collector"
            ] = pybamm.ZeroDimensionalSpatialMethod()
        elif self.options["dimensionality"] == 1:
            base_spatial_methods["current collector"] = pybamm.FiniteVolume()
        elif self.options["dimensionality"] == 2:
            base_spatial_methods["current collector"] = pybamm.ScikitFiniteElement()
        return base_spatial_methods

    @property
    def options(self):
        return self._options

    @options.setter
    def options(self, extra_options):
        default_options = {
            "operating mode": "current",
            "dimensionality": 0,
            "current collector": "uniform",
        }
        # Change the default for cell geometry based on which thermal option is provided
        extra_options = extra_options or {}

        options = pybamm.FuzzyDict(default_options)
        # any extra options overwrite the default options
        for name, opt in extra_options.items():
            if name in default_options:
                options[name] = opt
            else:
                raise pybamm.OptionError(
                    "Option '{}' not recognised. Best matches are {}".format(
                        name, options.get_best_matches(name)
                    )
                )

        # Some standard checks to make sure options are compatible
        if not (
            options["operating mode"] in ["current", "voltage", "power"]
            or callable(options["operating mode"])
        ):
            raise pybamm.OptionError(
                "operating mode '{}' not recognised".format(options["operating mode"])
            )

        if options["current collector"] not in [
            "uniform",
            "potential pair",
            "potential pair quite conductive",
        ]:
            raise pybamm.OptionError(
                "current collector model '{}' not recognised".format(
                    options["current collector"]
                )
            )
        if options["dimensionality"] not in [0, 1, 2]:
            raise pybamm.OptionError(
                "Dimension of current collectors must be 0, 1, or 2, not {}".format(
                    options["dimensionality"]
                )
            )

        if options["dimensionality"] == 0:
            if options["current collector"] not in ["uniform"]:
                raise pybamm.OptionError(
                    "current collector model must be uniform in 0D model"
                )

        self._options = options

    def set_standard_output_variables(self):
        # Time
        self.variables.update(
            {
                "Time": pybamm.t,
                "Time [s]": pybamm.t * self.timescale,
                "Time [min]": pybamm.t * self.timescale / 60,
                "Time [h]": pybamm.t * self.timescale / 3600,
            }
        )

        # Spatial
        L_y = self.param.L_y
        L_z = self.param.L_z
        var = pybamm.standard_spatial_vars
        if self.options["dimensionality"] == 1:
            self.variables.update({"z": var.z, "z [m]": var.z * L_z})
        elif self.options["dimensionality"] == 2:
            self.variables.update(
                {"y": var.y, "y [m]": var.y * L_y, "z": var.z, "z [m]": var.z * L_z}
            )

    def build_fundamental_and_external(self):
        # Get the fundamental variables
        for submodel_name, submodel in self.submodels.items():
            pybamm.logger.debug(
                "Getting fundamental variables for {} submodel ({})".format(
                    submodel_name, self.name
                )
            )
            self.variables.update(submodel.get_fundamental_variables())

        # Set any external variables to empty for now (may add later)
        self.external_variables = []
        self._built_fundamental_and_external = True

    def build_coupled_variables(self):
        # Note: pybamm will try to get the coupled variables for the submodels in the
        # order they are set by the user. If this fails for a particular submodel,
        # return to it later and try again. If setting coupled variables fails and
        # there are no more submodels to try, raise an error.
        submodels = list(self.submodels.keys())
        count = 0
        # For this part the FuzzyDict of variables is briefly converted back into a
        # normal dictionary for speed with KeyErrors
        self._variables = dict(self._variables)
        while len(submodels) > 0:
            count += 1
            for submodel_name, submodel in self.submodels.items():
                if submodel_name in submodels:
                    pybamm.logger.debug(
                        "Getting coupled variables for {} submodel ({})".format(
                            submodel_name, self.name
                        )
                    )
                    try:
                        self.variables.update(
                            submodel.get_coupled_variables(self.variables)
                        )
                        submodels.remove(submodel_name)
                    except KeyError as key:
                        if len(submodels) == 1 or count == 100:
                            # no more submodels to try
                            raise pybamm.ModelError(
                                "Missing variable for submodel '{}': {}.\n".format(
                                    submodel_name, key
                                )
                                + "Check the selected "
                                "submodels provide all of the required variables."
                            )
                        else:
                            # try setting coupled variables on next loop through
                            pybamm.logger.debug(
                                "Can't find {}, trying other submodels first".format(
                                    key
                                )
                            )
        # Convert variables back into FuzzyDict
        self._variables = pybamm.FuzzyDict(self._variables)

    def build_model_equations(self):
        # Set model equations
        for submodel_name, submodel in self.submodels.items():
            if submodel.external is False:
                pybamm.logger.debug(
                    "Setting rhs for {} submodel ({})".format(submodel_name, self.name)
                )

                submodel.set_rhs(self.variables)
                pybamm.logger.debug(
                    "Setting algebraic for {} submodel ({})".format(
                        submodel_name, self.name
                    )
                )

                submodel.set_algebraic(self.variables)
                pybamm.logger.debug(
                    "Setting boundary conditions for {} submodel ({})".format(
                        submodel_name, self.name
                    )
                )

                submodel.set_boundary_conditions(self.variables)
                pybamm.logger.debug(
                    "Setting initial conditions for {} submodel ({})".format(
                        submodel_name, self.name
                    )
                )
                submodel.set_initial_conditions(self.variables)
                submodel.set_events(self.variables)
                pybamm.logger.debug(
                    "Updating {} submodel ({})".format(submodel_name, self.name)
                )
                self.update(submodel)
                self.check_no_repeated_keys()

    def build_model(self):

        # Check if already built
        if self._built:
            raise pybamm.ModelError(
                """Model already built. If you are adding a new submodel, try using
                `model.update` instead."""
            )

        pybamm.logger.info("Building {}".format(self.name))

        if self._built_fundamental_and_external is False:
            self.build_fundamental_and_external()

        self.build_coupled_variables()

        self.build_model_equations()

        pybamm.logger.debug("Setting voltage variables ({})".format(self.name))
        self.set_voltage_variables()

        pybamm.logger.debug("Setting SoC variables ({})".format(self.name))
        self.set_soc_variables()

        self._built = True

    def new_copy(self, build=True):
        """
        Create a copy of the model. Overwrites the functionality of
        :class:`pybamm.BaseModel` to make sure that the submodels are updated correctly
        """
        # create without building
        # 'build' is not a keyword argument for the BaseBatteryModel class, but it
        # should be for all of the subclasses
        new_model = self.__class__(options=self.options, name=self.name, build=False)
        # update submodels
        new_model.submodels = self.submodels
        # clear submodel equations to avoid weird conflicts
        for submodel in self.submodels.values():
            submodel._rhs = {}
            submodel._algebraic = {}
            submodel._initial_conditions = {}
            submodel._boundary_conditions = {}
            submodel._variables = {}
            submodel._events = []

        # now build
        if build:
            new_model.build_model()
        new_model.use_jacobian = self.use_jacobian
        new_model.use_simplify = self.use_simplify
        new_model.convert_to_format = self.convert_to_format
        new_model.timescale = self.timescale
        new_model.length_scales = self.length_scales
        return new_model

    def set_external_circuit_submodel(self):
        """
        Define how the external circuit defines the boundary conditions for the model,
        e.g. (not necessarily constant-) current, voltage, etc
        """
        if self.options["operating mode"] == "current":
            self.submodels["external circuit"] = pybamm.external_circuit.CurrentControl(
                self.param
            )
        elif self.options["operating mode"] == "voltage":
            self.submodels[
                "external circuit"
            ] = pybamm.external_circuit.VoltageFunctionControl(self.param)
        elif self.options["operating mode"] == "power":
            self.submodels[
                "external circuit"
            ] = pybamm.external_circuit.PowerFunctionControl(self.param)
        elif callable(self.options["operating mode"]):
            self.submodels[
                "external circuit"
            ] = pybamm.external_circuit.FunctionControl(
                self.param, self.options["operating mode"]
            )

    def set_current_collector_submodel(self):

        if self.options["current collector"] in ["uniform"]:
            submodel = pybamm.current_collector.Uniform(self.param)
        elif self.options["current collector"] == "potential pair":
            if self.options["dimensionality"] == 1:
                submodel = pybamm.current_collector.PotentialPair1plus1D(self.param)
            elif self.options["dimensionality"] == 2:
                submodel = pybamm.current_collector.PotentialPair2plus1D(self.param)
        self.submodels["current collector"] = submodel

    def set_voltage_variables(self):

        ocv = 0
        ocv_dim = 0

        self.variables.update(
            {
                "Measured open circuit voltage": ocv,
                "Measured open circuit voltage [V]": ocv_dim,
            }
        )

        V_dim = self.variables["Terminal voltage [V]"]
        num_cells = pybamm.Parameter(
            "Number of cells connected in series to make a battery"
        )

        self.variables.update(
            {
                "Battery voltage [V]": V_dim * num_cells,
            }
        )

        # Cut-off voltage
        voltage = self.variables["Terminal voltage"]
        self.events.append(
            pybamm.Event(
                "Minimum voltage",
                voltage - self.param.voltage_low_cut,
                pybamm.EventType.TERMINATION,
            )
        )
        self.events.append(
            pybamm.Event(
                "Maximum voltage",
                voltage - self.param.voltage_high_cut,
                pybamm.EventType.TERMINATION,
            )
        )

        # Power
        I_dim = self.variables["Current [A]"]
        self.variables.update({"Terminal power [W]": I_dim * V_dim})

    def set_soc_variables(self):
        """
        Set variables relating to the state of charge.
        This function is overriden by the base ec models
        """
        pass

    def process_parameters_and_discretise(self, symbol, parameter_values, disc):
        """
        Process parameters and discretise a symbol using supplied parameter values
        and discretisation. Note: care should be taken if using spatial operators
        on dimensional symbols. Operators in pybamm are written in non-dimensional
        form, so may need to be scaled by the appropriate length scale. It is
        recommended to use this method on non-dimensional symbols.

        Parameters
        ----------
        symbol : :class:`pybamm.Symbol`
            Symbol to be processed
        parameter_values : :class:`pybamm.ParameterValues`
            The parameter values to use during processing
        disc : :class:`pybamm.Discretisation`
            The discrisation to use

        Returns
        -------
        :class:`pybamm.Symbol`
            Processed symbol
        """
        # Set y slices
        if disc.y_slices == {}:
            variables = list(self.rhs.keys()) + list(self.algebraic.keys())
            disc.set_variable_slices(variables)

        # Set boundary condtions (also requires setting parameter values)
        if disc.bcs == {}:
            self.boundary_conditions = parameter_values.process_boundary_conditions(
                self
            )
            disc.bcs = disc.process_boundary_conditions(self)

        # Process
        param_symbol = parameter_values.process_symbol(symbol)
        disc_symbol = disc.process_symbol(param_symbol)

        return disc_symbol
