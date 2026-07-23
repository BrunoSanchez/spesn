import iminuit
import numpy as np

from .satmodel import load_model


def compute_gaussian_likelihood(
    model,
    params,
    backoff,
    backoff_error,
):
    likelihood = np.nansum(
        (model(*params) - backoff) ** 2 / (2 * backoff_error**2)
    ) - np.nansum(np.log(backoff_error * np.sqrt(2 * np.pi)))
    return likelihood


class LikelihoodGaussian(object):
    def __init__(
        self,
        model_name,
        backoff,
        backoff_error,
        negative_log_likelihood,
    ):
        self.model = load_model(model_name)
        self.backoff = backoff
        self.backoff_error = backoff_error
        self.negative_log_likelihood = negative_log_likelihood

    def __call__(self, *params):
        log_likelihood = compute_gaussian_likelihood(
            self.model, params, self.backoff, self.backoff_error
        )
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
        cls, model_name, backoff, backoff_error, negative_log_likelihood=True
    ):
        likelihood = LikelihoodGaussian(
            model_name, backoff, backoff_error, negative_log_likelihood
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

        self.minuit = iminuit.Minuit(
            self.likelihood,
            init_params,
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
        model_name, backoff, backoff_error, negative_log_likelihood
    )
    if minuit_parameters is not None:
        fitter.minuit_parameters = minuit_parameters
    fitter.init_minuit_fit(init_params)
    result = fitter.run(migrad=migrad, hesse=hesse, minos=minos, n_iter=n_iter)
    return result


def launch_several_fits_mp(
    model_name,
    backoff_list,
    backoff_error_list,
    init_params_list,
    minuit_parameters=None,
    negative_log_likelihood=True,
    migrad=True,
    hesse=False,
    minos=False,
    n_iter=1,
):
    from multiprocessing import Pool

    def fit_single(args):
        backoff, backoff_error, init_params = args
        return launch_fit(
            model_name,
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

    with Pool() as pool:
        results = pool.map(
            fit_single, zip(backoff_list, backoff_error_list, init_params_list)
        )

    return results
