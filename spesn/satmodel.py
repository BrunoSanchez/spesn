
import numpy as np





def load_model(model_name):
    """
    Load a model function based on the provided model name.

    Parameters:
    model_name : str
        Name of the model to load.

    Returns:
    function
        The corresponding model function.
    """
    if model_name == "broken_line":
        return broken_line_model
    else:
        raise ValueError(f"Model '{model_name}' is not recognized.")


def broken_line_model(x, x0, y0,slope):
    """
    Broken line model function.

    Parameters:
    x : array-like
        Independent variable.
    x0 : float
        Breakpoint in the independent variable.
    y0 : float
        Value of the dependent variable at the breakpoint.
    slope : float
        Slope of the line after the breakpoint.

    Returns:
    array-like
        Dependent variable values corresponding to the input x.
    """
    return np.where(x < x0, y0 + slope * (x - x0), y0)



