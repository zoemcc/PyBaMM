#
# Class for interactive plotting of variables from models
#
import pybamm
import warnings


class InteractivePlot(pybamm.QuickPlot):
    """
    An interactive plot which can be updated with new parameter values. This is called
    via :meth:`pybamm.Simulation.interactive`: note that the parameter values to be made
    interactive must be specified in advance.

    Parameters
    ----------
    sim : :class:`pybamm.Simulation`
        The simulation to be used for plotting
    hold : int
        How many previous solutions to hold. Set to -1 to hold all previous solutions.
    kwargs : keyword-arguments
        Keyword arguments to be passed to :class:`pybamm.QuickPlot`

    **Extends**: :class:`pybamm.QuickPlot`
    """

    def __init__(self, sim, hold=1, **kwargs):
        super().__init__(sim.solution, **kwargs)
        self.hold = hold

    def dynamic_plot(self, testing=False):
        """
        Generate a dynamic plot with a slider to control the time, and text boxes to
        control the inputs. We recommend using ipywidgets instead of this function if
        you are using jupyter notebooks
        """

        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider, TextBox, Button

        # create an initial plot at time 0
        self.plot(0)

        # Slider
        axcolor = "lightgoldenrodyellow"
        ax_slider = plt.axes([0.315, 0.02, 0.37, 0.03], facecolor=axcolor)
        self.slider = Slider(ax_slider, "Time", 0, self.max_t, valinit=0)
        self.slider.on_changed(self.update)

        # Text boxes
        self.text_boxes = {}
        for input_name, value in self.initial_inputs.items():
            ax_box = plt.axes([0.9, 0.02, 0.98, 0.03])
            self.text_boxes[input_name] = TextBox(
                ax_box, input_name, initial=str(value)
            )
            self.text_boxes[input_name].on_submit(SubmitText(self, input_name))

        # Reset button
        ax_reset = plt.axes([0.9, 0.02, 0.98, 0.03])
        self.reset_button = Button(ax_reset, "Reset")
        self.reset_button.on_clicked(self.reset)

        # Undo button
        ax_undo = plt.axes([0.9, 0.02, 0.98, 0.03])
        self.undo_button = Button(ax_undo, "Undo")
        self.undo_button.on_clicked(self.undo)

        # ignore the warning about tight layout
        warnings.simplefilter("ignore")
        self.fig.tight_layout()
        warnings.simplefilter("always")

        if not testing:  # pragma: no cover
            plt.show()

    def reset(self):
        "Reset the interactive plot"

    def undo(self):
        "Undo the last action"


class SubmitText(object):
    def __init__(self, plot, input_name):
        self.plot = plot
        self.input_name = input_name

    def __call__(self, value):
        self.plot.inputs[self.input_name] = value
        # Run the simulation with new inputs
        # Query self.hold
        # Update legend
        # Update external file with the parameter values
        # Update text on screen (success, failure, see terminal for more detail)
