"""Saturation "back-off" models for bright-star photometry.

Each model returns the magnitude back-off ``dm = m_meas - m_ref`` as a
function of the reference (true) magnitude ``x = m_ref``.  Convention:
``dm`` is ~0 for faint stars and rises for bright (low-magnitude) stars,
because saturated stars are measured too faint.

Three physical regimes drive the back-off (pick the model by which one
your detector is in):

  * A/D clip well below full-well  -> counts above the ADC level are
    *lost* -> strong logarithmic back-off (``gaussian_clip_model``,
    ``moffat_clip_model``).
  * Full-well reached with bleeding -> charge is *conserved* and spills
    along the column; back-off appears only from trail flux missed by the
    aperture (``bleed_residual_model``).
  * Soft nonlinearity below saturation -> charge sharing between pixels
    compresses the response starting ~1/3 full-well
    (``quadratic_nonlinearity_model``).

Key references (see individual functions for the specific one used):

  * Maiz-Apellaniz 2002, "Photometry of Saturated Stars in CCD Images",
    2002 HST Calibration Workshop, STScI.
    https://www.stsci.edu/largefiles/hst/HST_overview/documents/calworkshop/workshop2002/CW2002_Papers/jmaiz.pdf
  * Ma et al. 2014, "The nonlinear photon transfer curve of CCDs and its
    effects on photometry", arXiv:1407.8280.
  * Kochanek et al. 2017, "The ASAS-SN Light Curve Server v1.0",
    PASP 129, 104502 (bleed-trail redistribution; Pojmanski 2002 heritage).
    https://iopscience.iop.org/article/10.1088/1538-3873/aa80d9
  * Gilliland 1994; ACS beyond-saturation linearity (charge conservation
    when full-well ~ A/D level).
  * LSST Community, "CCD Nonlinearity Near Saturation".
    https://community.lsst.org/t/ccd-nonlinearity-near-saturation/2288
  * Photometry of Saturated Stars with Neural Networks, arXiv:2404.15405
    (modern PSF-wing + bleed context).
"""

import numpy as np

# 2.5 / ln(10): converts a natural-log flux ratio to a magnitude.
_POGSON = 2.5 / np.log(10.0)  # = 1.0857362...


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
    models = {
        "broken_line": broken_line_model,
        "gaussian_clip": gaussian_clip_model,
        "moffat_clip": moffat_clip_model,
        "quadratic_nonlinearity": quadratic_nonlinearity_model,
        "bleed_residual": bleed_residual_model,
        "softplus": softplus_model,
        "logistic_flux": logistic_flux_model,
        "quadratic_plus_gaussian_clip": quadratic_plus_gaussian_clip_model,
    }
    try:
        return models[model_name]
    except KeyError:
        raise ValueError(
            f"Model '{model_name}' is not recognized. "
            f"Available models: {sorted(models)}"
        )


def list_models():
    """Return the list of registered model names."""
    return [
        "broken_line",
        "gaussian_clip",
        "moffat_clip",
        "quadratic_nonlinearity",
        "bleed_residual",
        "softplus",
        "logistic_flux",
        "quadratic_plus_gaussian_clip",
    ]


def broken_line_model(x, x0, y0, slope):
    """
    Broken line model function.

    Empirical two-slope response: flat (``y0``) for faint stars, linear for
    bright stars.  The simplest phenomenological back-off; a piecewise-linear
    response is a standard high-dynamic-range CCD approximation near
    saturation (see LSST Community, "CCD Nonlinearity Near Saturation",
    https://community.lsst.org/t/ccd-nonlinearity-near-saturation/2288).

    Parameters:
    x : array-like
        Independent variable (reference magnitude ``m_ref``).
    x0 : float
        Breakpoint in the independent variable (saturation onset magnitude).
    y0 : float
        Value of the dependent variable at the breakpoint.
    slope : float
        Slope of the line before the breakpoint (bright side).

    Returns:
    array-like
        Magnitude back-off ``dm = m_meas - m_ref``.
    """
    return np.where(x < x0, y0 + slope * (x - x0), y0)


def gaussian_clip_model(x, m_sat, dm0=0.0):
    r"""
    Hard-clip flux-loss back-off for a Gaussian PSF (the core model).

    First-principles model for the A/D-clip regime, where the ADC saturation
    level sits well below the pixel full-well so charge above the clip is
    *lost* rather than bled (Maiz-Apellaniz 2002).  For a 2D Gaussian PSF of
    total flux ``F`` clipped per-pixel at level ``S``, integrating the flux
    above the clip gives the closed form::

        F_sat   = 2 * pi * sigma**2 * S           # onset flux
        x_over  = ln(F / F_sat)                    # >= 0 past onset
        F_meas  = F_sat * (1 + x_over)             # measured flux caps ~log

    so the measured flux grows only logarithmically past onset.  In
    magnitudes, with ``u = (m_sat - m_ref) / 1.0857``::

        dm = 1.0857 * u - 2.5 * log10(1 + u)       (bright side)

    Reference:
        Maiz-Apellaniz 2002, "Photometry of Saturated Stars in CCD Images",
        2002 HST Calibration Workshop, STScI.
        https://www.stsci.edu/largefiles/hst/HST_overview/documents/calworkshop/workshop2002/CW2002_Papers/jmaiz.pdf

    Parameters:
    x : array-like
        Reference magnitude ``m_ref``.
    m_sat : float
        Saturation onset magnitude (the single physical parameter);
        ``m_sat = ZP - 2.5 * log10(2 * pi * sigma**2 * S)``.
    dm0 : float, optional
        Constant baseline offset (e.g. residual zero-point). Default 0.

    Returns:
    array-like
        Magnitude back-off ``dm = m_meas - m_ref``.
    """
    x = np.asarray(x, dtype=float)
    u = (m_sat - x) / _POGSON  # = ln(F / F_sat)
    u_safe = np.where(x < m_sat, u, 0.0)  # keep 1 + u >= 1 on the faint side
    dm_bright = _POGSON * u_safe - 2.5 * np.log10(1.0 + u_safe)
    return np.where(x < m_sat, dm_bright, 0.0) + dm0


def moffat_clip_model(x, m_sat, beta, dm0=0.0):
    r"""
    Hard-clip flux-loss back-off for a Moffat PSF (realistic wings).

    Same A/D-clip physics as :func:`gaussian_clip_model` but with a Moffat
    profile ``I(r) = I0 * [1 + (r/alpha)**2]**(-beta)``, which has fatter
    wings than a Gaussian and therefore a gentler, slower back-off.
    Integrating the flux above the per-pixel clip yields a closed form
    (``rho = F / F_sat``)::

        F_meas / F_sat = beta * rho**(1/beta) - (beta - 1)

    which reduces to the Gaussian result ``1 + ln(rho)`` as ``beta -> inf``.
    In magnitudes, with ``rho = 10**(0.4 * (m_sat - m_ref))``::

        dm = -2.5 * log10( (beta * rho**(1/beta) - (beta - 1)) / rho )

    The brighter-fatter effect (PSF broadening with flux) means ``alpha`` and
    hence the effective wing shape grow with brightness; see Ma et al. 2014.

    References:
        Maiz-Apellaniz 2002, STScI 2002 HST Calibration Workshop
        (A/D-clip flux loss).
        https://www.stsci.edu/largefiles/hst/HST_overview/documents/calworkshop/workshop2002/CW2002_Papers/jmaiz.pdf
        Ma et al. 2014, "The nonlinear photon transfer curve of CCDs and its
        effects on photometry", arXiv:1407.8280 (brighter-fatter PSF).

    Parameters:
    x : array-like
        Reference magnitude ``m_ref``.
    m_sat : float
        Saturation onset magnitude.
    beta : float
        Moffat beta index (typ. 2.5-4.5). Larger beta -> more Gaussian.
    dm0 : float, optional
        Constant baseline offset. Default 0.

    Returns:
    array-like
        Magnitude back-off ``dm = m_meas - m_ref``.
    """
    x = np.asarray(x, dtype=float)
    rho = np.power(10.0, 0.4 * (m_sat - x))  # F / F_sat
    rho_safe = np.where(x < m_sat, rho, 1.0)  # keep the log argument positive
    fmeas_over_fsat = beta * np.power(rho_safe, 1.0 / beta) - (beta - 1.0)
    dm_bright = -2.5 * np.log10(fmeas_over_fsat / rho_safe)
    return np.where(x < m_sat, dm_bright, 0.0) + dm0


def quadratic_nonlinearity_model(x, zeropoint, beta_nl, dm0=0.0):
    r"""
    Soft (sub-saturation) nonlinearity back-off, quadratic response.

    Before any pixel reaches full-well, charge sharing between pixels already
    compresses the response.  The photon transfer curve becomes nonlinear at
    about 1/3 full-well and is well described by a quadratic in the collected
    charge (Ma et al. 2014)::

        F_meas = F - beta_nl * F**2 = F * (1 - beta_nl * F)
        dm     = -2.5 * log10(1 - beta_nl * F)

    with ``F = 10**(-0.4 * (m_ref - zeropoint))``.  Use as the gentle,
    smoothly-accelerating term that precedes the sharp hard-clip knee.

    References:
        Ma et al. 2014, "The nonlinear photon transfer curve of CCDs and its
        effects on photometry", arXiv:1407.8280.
        Antilogus et al. 2014 (brighter-fatter / charge sharing).

    Parameters:
    x : array-like
        Reference magnitude ``m_ref``.
    zeropoint : float
        Photometric zero-point mapping magnitude to flux units of ``beta_nl``.
    beta_nl : float
        Nonlinearity coefficient (small, > 0), set by the charge-sharing rate.
    dm0 : float, optional
        Constant baseline offset. Default 0.

    Returns:
    array-like
        Magnitude back-off ``dm = m_meas - m_ref``.
    """
    x = np.asarray(x, dtype=float)
    flux = np.power(10.0, -0.4 * (x - zeropoint))
    arg = np.clip(1.0 - beta_nl * flux, 1e-12, None)  # guard the log
    return -2.5 * np.log10(arg) + dm0


def bleed_residual_model(x, m_sat, k, dm0=0.0):
    r"""
    Charge-conserving bleed regime with imperfect aperture recovery.

    When the full-well is close to the A/D level the charge is *conserved*
    and simply bleeds along the column, so summing an aperture that captures
    the bleed trail recovers the true flux (back-off ~0).  A residual
    back-off appears only when a fixed aperture misses part of a bleed trail
    whose length grows with the excess flux above onset.  Modelling the
    escaping fraction as ``k`` times the over-full fraction
    ``(F - F_sat)/F``::

        frac_over = 1 - 10**(-0.4 * (m_sat - m_ref))
        dm        = -2.5 * log10(1 - k * frac_over)

    ASAS / ASAS-SN instead redistribute bleed-trail flux back to the centre
    (Pojmanski 2002; Kochanek et al. 2017), which removes this residual.

    References:
        Kochanek et al. 2017, "The ASAS-SN Light Curve Server v1.0",
        PASP 129, 104502 (bleed-trail flux redistribution).
        https://iopscience.iop.org/article/10.1088/1538-3873/aa80d9
        Gilliland 1994; ACS beyond-saturation linearity via charge
        conservation when full-well ~ A/D level.

    Parameters:
    x : array-like
        Reference magnitude ``m_ref``.
    m_sat : float
        Saturation onset magnitude.
    k : float
        Fraction of over-full charge that escapes the aperture (0 = perfect
        bleed recovery, 1 = all excess lost).
    dm0 : float, optional
        Constant baseline offset. Default 0.

    Returns:
    array-like
        Magnitude back-off ``dm = m_meas - m_ref``.
    """
    x = np.asarray(x, dtype=float)
    frac_over = np.clip(1.0 - np.power(10.0, -0.4 * (m_sat - x)), 0.0, None)
    arg = np.clip(1.0 - k * frac_over, 1e-12, None)  # guard the log
    return np.where(x < m_sat, -2.5 * np.log10(arg), 0.0) + dm0


def softplus_model(x, m_sat, a, s, dm0=0.0):
    r"""
    Smooth (softplus) empirical back-off with a differentiable knee.

    A robust smoothing of the broken-line / hard-clip knee, useful as a
    single fitter across the transition::

        dm = a * s * log(1 + exp((m_sat - m_ref) / s)) + dm0

    ``m_sat`` sets the knee location, ``a`` the asymptotic bright-side slope,
    and ``s`` the knee sharpness (``s -> 0`` recovers a hard hinge).  Computed
    with a numerically stable log-sum-exp.  Phenomenological, but anchors to
    the piecewise CCD response near saturation (LSST Community).

    Reference:
        LSST Community, "CCD Nonlinearity Near Saturation".
        https://community.lsst.org/t/ccd-nonlinearity-near-saturation/2288

    Parameters:
    x : array-like
        Reference magnitude ``m_ref``.
    m_sat : float
        Knee (saturation onset) magnitude.
    a : float
        Asymptotic bright-side slope of the back-off.
    s : float
        Knee sharpness (magnitudes); smaller is sharper.
    dm0 : float, optional
        Constant baseline offset. Default 0.

    Returns:
    array-like
        Magnitude back-off ``dm = m_meas - m_ref``.
    """
    x = np.asarray(x, dtype=float)
    z = (m_sat - x) / s
    softplus = np.logaddexp(0.0, z)  # = log(1 + exp(z)), overflow-safe
    return a * s * softplus + dm0


def logistic_flux_model(x, zeropoint, flux_max, kk, dm0=0.0):
    r"""
    Smoothly-saturating (logistic-type) detector flux response.

    Models a bounded detector response that saturates toward a ceiling
    ``flux_max`` (~ full-well per pixel times the effective number of
    saturated pixels)::

        F_meas = flux_max * (1 - exp(-kk * F / flux_max))
        dm     = -2.5 * log10(F_meas / F)

    with ``F = 10**(-0.4 * (m_ref - zeropoint))``.  Linear for faint stars,
    diverging back-off for bright ones as ``F_meas -> flux_max``.

    Reference:
        LSST Community, "CCD Nonlinearity Near Saturation" (bounded/
        piecewise response near full-well).
        https://community.lsst.org/t/ccd-nonlinearity-near-saturation/2288

    Parameters:
    x : array-like
        Reference magnitude ``m_ref``.
    zeropoint : float
        Photometric zero-point mapping magnitude to flux units.
    flux_max : float
        Response ceiling in the same flux units.
    kk : float
        Small-signal response slope (dimensionless, ~1 for a linear onset).
    dm0 : float, optional
        Constant baseline offset. Default 0.

    Returns:
    array-like
        Magnitude back-off ``dm = m_meas - m_ref``.
    """
    x = np.asarray(x, dtype=float)
    flux = np.power(10.0, -0.4 * (x - zeropoint))
    fmeas = flux_max * (1.0 - np.exp(-kk * flux / flux_max))
    ratio = np.clip(fmeas / flux, 1e-12, None)  # guard the log
    return -2.5 * np.log10(ratio) + dm0


def quadratic_plus_gaussian_clip_model(x, m_sat, zeropoint, beta_nl, dm0=0.0):
    r"""
    Composite: soft quadratic nonlinearity + hard-clip Gaussian back-off.

    Recommended full-curve model.  Sums the gentle sub-saturation
    nonlinearity (:func:`quadratic_nonlinearity_model`, Ma et al. 2014) that
    curves the response before the knee, and the sharp A/D-clip back-off
    (:func:`gaussian_clip_model`, Maiz-Apellaniz 2002) that dominates past
    onset::

        dm = quadratic_nonlinearity(m_ref) + gaussian_clip(m_ref)

    References:
        Maiz-Apellaniz 2002, STScI 2002 HST Calibration Workshop.
        https://www.stsci.edu/largefiles/hst/HST_overview/documents/calworkshop/workshop2002/CW2002_Papers/jmaiz.pdf
        Ma et al. 2014, arXiv:1407.8280.

    Parameters:
    x : array-like
        Reference magnitude ``m_ref``.
    m_sat : float
        Saturation onset magnitude (hard-clip term).
    zeropoint : float
        Photometric zero-point (quadratic term).
    beta_nl : float
        Nonlinearity coefficient (quadratic term).
    dm0 : float, optional
        Constant baseline offset. Default 0.

    Returns:
    array-like
        Magnitude back-off ``dm = m_meas - m_ref``.
    """
    soft = quadratic_nonlinearity_model(x, zeropoint, beta_nl, dm0=0.0)
    hard = gaussian_clip_model(x, m_sat, dm0=0.0)
    return soft + hard + dm0


def plot_models(x=None, models=None, data=None, ax=None, figsize=(12, 5)):
    """
    Two-panel comparison plot of the saturation back-off models.

    Left panel:  back-off ``dm = m_meas - m_ref`` versus reference magnitude.
    Right panel: measured magnitude ``m_meas`` versus reference magnitude,
                 with the 1:1 (no-saturation) line for reference.

    Parameters:
    x : array-like, optional
        Reference-magnitude grid. Defaults to ``np.linspace(6, 14, 400)``.
    models : dict, optional
        Mapping ``label -> (model_name_or_func, params_dict)``. Defaults to a
        representative demonstration set sharing a common onset magnitude.
    data : tuple, optional
        ``(m_ref, m_meas)`` observed points to overplot (scatter).
    ax : sequence of 2 matplotlib Axes, optional
        Axes to draw into. If None, a new figure with two panels is created.
    figsize : tuple, optional
        Figure size when creating a new figure.

    Returns:
    (fig, (ax_left, ax_right))
        The Matplotlib figure and the two axes.
    """
    import matplotlib.pyplot as plt

    if x is None:
        x = np.linspace(6.0, 14.0, 400)
    x = np.asarray(x, dtype=float)

    if models is None:
        # Representative parameters sharing a common onset m_sat = 10.5.
        m_sat = 10.5
        zeropoint = 20.0
        models = {
            "broken_line": ("broken_line", dict(x0=m_sat, y0=0.0, slope=-0.3)),
            "gaussian_clip": ("gaussian_clip", dict(m_sat=m_sat)),
            "moffat_clip (beta=3)": (
                "moffat_clip",
                dict(m_sat=m_sat, beta=3.0),
            ),
            "quadratic_nonlinearity": (
                "quadratic_nonlinearity",
                dict(zeropoint=zeropoint, beta_nl=2e-6),
            ),
            "bleed_residual (k=0.5)": (
                "bleed_residual",
                dict(m_sat=m_sat, k=0.5),
            ),
            "softplus": ("softplus", dict(m_sat=m_sat, a=0.35, s=0.3)),
            "logistic_flux": (
                "logistic_flux",
                dict(zeropoint=zeropoint, flux_max=8e3, kk=1.0),
            ),
            "quad+gaussian_clip": (
                "quadratic_plus_gaussian_clip",
                dict(m_sat=m_sat, zeropoint=zeropoint, beta_nl=2e-6),
            ),
        }

    if ax is None:
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=figsize)
    else:
        ax_l, ax_r = ax
        fig = ax_l.figure

    for label, (model, params) in models.items():
        func = load_model(model) if isinstance(model, str) else model
        dm = func(x, **params)
        ax_l.plot(x, dm, label=label, lw=1.8)
        ax_r.plot(x, x + dm, label=label, lw=1.8)

    if data is not None:
        m_ref, m_meas = np.asarray(data[0]), np.asarray(data[1])
        ax_l.scatter(
            m_ref,
            m_meas - m_ref,
            s=10,
            c="k",
            alpha=0.5,
            label="data",
            zorder=5,
        )
        ax_r.scatter(m_ref, m_meas, s=10, c="k", alpha=0.5, label="data", zorder=5)

    # 1:1 reference line on the right panel.
    lims = [float(x.min()), float(x.max())]
    ax_r.plot(lims, lims, "k--", lw=1.0, alpha=0.6, label="1:1 (no saturation)")

    ax_l.axhline(0.0, color="k", lw=0.8, alpha=0.4)
    ax_l.set_xlabel("reference magnitude  m_ref")
    ax_l.set_ylabel("back-off  dm = m_meas - m_ref")
    ax_l.set_title("Saturation back-off")
    ax_l.legend(fontsize=8, loc="upper right")

    ax_r.set_xlabel("reference magnitude  m_ref")
    ax_r.set_ylabel("measured magnitude  m_meas")
    ax_r.set_title("Measured vs reference")
    ax_r.legend(fontsize=8, loc="upper left")

    # Brighter stars (small magnitude) on the left of each panel.
    ax_l.invert_xaxis()
    ax_r.invert_xaxis()
    ax_r.invert_yaxis()

    fig.tight_layout()
    return fig, (ax_l, ax_r)
