

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
    return np.where(x < x0, y0 + slope * (x - x0), y0 )