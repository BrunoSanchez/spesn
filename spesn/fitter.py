import iminuit
import numpy as np
import multiprocessing as mp
from iminuit.util import describe

from .satmodel import load_model


def compute_gaussian_likelihood(
    model,
    params,
    magnitude,
    backoff,
    backoff_error,
):
    """Gaussian log-likelihood of the back-off data given the model.

    Returns log L = -sum[ (model - backoff)^2 / (2 sigma^2) ]
                    - sum[ log(sigma sqrt(2 pi)) ].
    """
    residual = model(magnitude, *params) - backoff
    log_likelihood = -np.nansum(
        residual**2 / (2 * backoff_error**2)
    ) - np.nansum(np.log(backoff_error * np.sqrt(2 * np.pi)))
    return log_likelihood


class LikelihoodGaussian(object):
    def __init__(
        self,
        model_name,
        magnitude,
        backoff,
        backoff_error,
        negative_log_likelihood,
    ):
        self.model = load_model(model_name)
        self.magnitude = magnitude
        self.backoff = backoff
        self.backoff_error = backoff_error
        self.negative_log_likelihood = negative_log_likelihood
        # Model parameter names, excluding the first arg (the magnitude).
        # iminuit cannot introspect the ``*params`` signature of ``__call__``,
        # so these are passed to Minuit explicitly via ``name=``.
        self.parameter_names = list(describe(self.model))[1:]

    def __call__(self, *params):
        log_likelihood = compute_gaussian_likelihood(
            self.model, params, self.magnitude, self.backoff, self.backoff_error
        )
        # Minuit minimises with errordef=LIKELIHOOD, so it needs the
        # NEGATIVE log-likelihood.
        if self.negative_log_likelihood:
            return -log_likelihood
        else:
            return log_likelihood


class Fitter(object):
    def __init__(
        self,
        likelihood,
        minuit_parameters=(),
    ):
        self.likelihood = likelihood
        self.minuit_parameters = minuit_parameters

    @classmethod
    def init_from_data(
        cls, model_name, magnitude, backoff, backoff_error, negative_log_likelihood=True,
    ):
        likelihood = LikelihoodGaussian(
            model_name, magnitude, backoff, backoff_error, negative_log_likelihood
        )
        return cls(likelihood)

    def setup_minuit(
        self,
    ):

        self.minuit.errordef = iminuit.Minuit.LIKELIHOOD
        for parameters in self.minuit_parameters:
            d = self.minuit_parameters[parameters]
            self.minuit.errors[parameters] = d["error"] if "error" in d else d["value"]
            self.minuit.errors[parameters] = (
                0 if "fixed" in d and d["fixed"] else self.minuit.errors[parameters]
            )
            self.minuit.fixed[parameters] = d["fixed"] if "fixed" in d else False
            limit_low = d["limit_low"] if "limit_low" in d else None
            limit_up = d["limit_up"] if "limit_up" in d else None
            self.minuit.limits[parameters] = (limit_low, limit_up)

    def init_minuit_fit(self, init_params):
        names = self.likelihood.parameter_names
        # Accept a dict {name: value} or an ordered sequence of start values.
        if isinstance(init_params, dict):
            init_values = [init_params[name] for name in names]
        else:
            init_values = list(init_params)
        if len(init_values) != len(names):
            raise ValueError(
                f"init_params has {len(init_values)} values but model "
                f"expects {len(names)} parameters: {names}"
            )
        # Pass names explicitly: Minuit cannot infer them from the
        # variadic ``*params`` signature of the likelihood.
        self.minuit = iminuit.Minuit(
            self.likelihood,
            *init_values,
            name=names,
        )
        self.setup_minuit()

    def run(self, migrad=True, hesse=False, minos=False, n_iter=1):
        """Run Minuit optimization and return fitted parameter values.

        Args:
            migrad (bool): Run MIGRAD algorithm.
            hesse (bool): Compute HESSE errors.
            minos (bool): Compute MINOS intervals (may be slow).
            n_iter (int): Number of MIGRAD iterations to perform.

        Returns:
            dict: Fitted parameter values.
        """
        if migrad:
            for i in range(n_iter):
                if n_iter != 1:
                    print(f"Iteration {i+1}/{n_iter}\n")
                print(self.minuit.migrad())
        if hesse:
            print(self.minuit.hesse())
        if minos:
            try:
                print(self.minuit.minos())
            except RuntimeError:
                pass

        return self.minuit.values.to_dict()


def launch_fit(
    model_name,
    magnitude,
    backoff,
    backoff_error,
    init_params,
    minuit_parameters=None,
    negative_log_likelihood=True,
    migrad=True,
    hesse=False,
    minos=False,
    n_iter=1,
):
    fitter = Fitter.init_from_data(
        model_name, magnitude, backoff, backoff_error, negative_log_likelihood
    )
    if minuit_parameters is not None:
        fitter.minuit_parameters = minuit_parameters
    fitter.init_minuit_fit(init_params)
    result = fitter.run(migrad=migrad, hesse=hesse, minos=minos, n_iter=n_iter)
    return result



def launch_several_fits_multiprocessing(
    model_name,
    magnitude_list,
    backoff_list,
    backoff_error_list,
    init_params_list,
    number_threads=1,
    minuit_parameters=None,
    negative_log_likelihood=True,
    migrad=True,
    hesse=False,
    minos=False,
    n_iter=1,
    
):

    def fit_single(args):
        magnitude, backoff, backoff_error, init_params = args
        return launch_fit(
            model_name,
            magnitude,
            backoff,
            backoff_error,
            init_params,
            minuit_parameters=minuit_parameters,
            negative_log_likelihood=negative_log_likelihood,
            migrad=migrad,
            hesse=hesse,
            minos=minos,
            n_iter=n_iter,
        )

    if number_threads == 1:
        results = [fit_single(args) for args in zip(magnitude_list, backoff_list, backoff_error_list, init_params_list)]
    else:
        with mp.Pool(number_threads) as pool:
            results = pool.map(
                fit_single, zip(magnitude_list, backoff_list, backoff_error_list, init_params_list)
            )

    return results
