from .type import Gradient, Hessian, ParetoMixtureParameters, Sample
import numpy as np
from collections import defaultdict
import pandas as pd
from numpy.linalg import inv

def fit_pareto_mixture_parameters(sample: Sample, k: int | None = None) -> tuple[ParetoMixtureParameters, pd.DataFrame]:
    """
    A function that fits alpha, beta and p on a sample with a given k.
    If k is None, then k = len(sample)
    """
    if k is None:
        k = len(sample.data)

    subsample = sample.data[:k]
    subsample = np.array(sorted(subsample))
    
    # Initial parameters, should converge anyways
    current_parameters = ParetoMixtureParameters(alpha=2.0, beta=1.0, p=0.8)

    subsample = Sample(subsample[:k])

    MAX_ITERS_WITHOUT_PROGRESS = 8
    MAX_ITERS = 300

    last_ll = float("inf")
    num_iters_without_progress = 0
    STEP_SIZE = 0.9
    history = defaultdict(dict[str, object]) # we keep track of the history of the parameters for debug purposes
    for step in range(1, MAX_ITERS):
        ll, grad, hess = loglikelihood(current_parameters, subsample), gradient(current_parameters, subsample), hessian(current_parameters, subsample)
        history[step]["params"] = current_parameters
        history[step]["ll"] = ll
        history[step]["grad"] = grad
        history[step]["hess"] = hess
        if ll < last_ll + 0.00001:
            num_iters_without_progress += 1
            if num_iters_without_progress >= MAX_ITERS_WITHOUT_PROGRESS:
                return current_parameters, pd.DataFrame(history).T
        else:
            num_iters_without_progress = 0
        last_ll = ll

        dalpha, dbeta, dp = hess_inv_gradient(hess, grad)
        current_parameters.alpha -= dalpha * STEP_SIZE
        current_parameters.beta -= dbeta * STEP_SIZE
        current_parameters.p -= dp * STEP_SIZE
        current_parameters = move_to_feas_region(current_parameters)
    raise ValueError(f"Did not converge in {MAX_ITERS} iterations")

def move_to_feas_region(pm: ParetoMixtureParameters) -> ParetoMixtureParameters:
    # we make sure p > 0.5 w.l.g.
    if pm.p < 0.5:
        pm.p, pm.alpha, pm.beta = 1 - pm.p, pm.alpha + pm.beta, -pm.beta
    if pm.p > 2.0:
        pm.p = 2.0

    # Lower bounds on alpha and alpha+beta
    if pm.alpha < 0.0001:
        pm.alpha = 0.0001
    if pm.alpha + pm.beta < 0.0001:
        pm.beta = 0.0001 - pm.alpha
    if np.abs(pm.beta) < 0.001:
        pm.beta = pm.beta * 0.3 / np.abs(pm.beta)
    return pm

def hess_inv_gradient(hess: Hessian, grad: Gradient):
    """
    Returns the inverse of the hessian times the gradient
    """
    hess_m = np.array(
        [
            [hess.dll_dalpha2, hess.dll_dalphadbeta, hess.dll_dalphadp],
            [hess.dll_dalphadbeta, hess.dll_dbeta2, hess.dll_dbetadp],
            [hess.dll_dalphadp, hess.dll_dbetadp, hess.dll_dp2],
        ]
    )
    grad_v = np.array(
        [grad.dll_dalpha, grad.dll_dbeta, grad.dll_dp]
    )
    return inv(hess_m) @ grad_v

def loglikelihood(pm: ParetoMixtureParameters, sample: Sample) -> float:
    """
    Returns the loglikelihood of the sample given the parameters
    """
    alpha = pm.alpha
    beta = pm.beta
    p = pm.p
    x = sample.data
    return np.sum(
        np.log(p * x ** (-alpha - 1.) + (1 - p) * x ** (-alpha - beta - 1.))
    )

def gradient(pm: ParetoMixtureParameters, sample: Sample) -> Gradient:
    """
    Returns the gradient of the loglikelihood of the sample given the parameters
    """
    alpha = pm.alpha
    beta = pm.beta
    p = pm.p
    x = sample.data
    dll_dalpha = np.sum(
        (-alpha*p*x**(-alpha - 1)*np.log(x) + p*x**(-alpha - 1) + x**(-alpha - beta - 1)*(1 - p)*(-alpha - beta)*np.log(x) + x**(-alpha - beta - 1)*(1 - p))/(alpha*p*x**(-alpha - 1) - x**(-alpha - beta - 1)*(1 - p)*(-alpha - beta))
    )
    dll_dbeta = np.sum(
        (x**(-alpha - beta - 1)*(1 - p)*(-alpha - beta)*np.log(x) + x**(-alpha - beta - 1)*(1 - p))/(alpha*p*x**(-alpha - 1) - x**(-alpha - beta - 1)*(1 - p)*(-alpha - beta))
    )
    dll_dp = np.sum(
        (alpha*x**(-alpha - 1) + x**(-alpha - beta - 1)*(-alpha - beta))/(alpha*p*x**(-alpha - 1) - x**(-alpha - beta - 1)*(1 - p)*(-alpha - beta))
    )
    return Gradient(dll_dalpha, dll_dbeta, dll_dp)


def hessian(pm: ParetoMixtureParameters, sample: Sample) -> Hessian:
    """
    Returns the hessian of the loglikelihood of the sample given the parameters
    """
    alpha = pm.alpha
    beta = pm.beta
    p = pm.p
    x = sample.data
    dll_dalpha2 = np.sum(
        (alpha*p*x**(-alpha - 1)*np.log(x)**2 - 2*p*x**(-alpha - 1)*np.log(x) - x**(-alpha - beta - 1)*(1 - p)*(-alpha - beta)*np.log(x)**2 - 2*x**(-alpha - beta - 1)*(1 - p)*np.log(x))/(alpha*p*x**(-alpha - 1) - x**(-alpha - beta - 1)*(1 - p)*(-alpha - beta)) + (-alpha*p*x**(-alpha - 1)*np.log(x) + p*x**(-alpha - 1) + x**(-alpha - beta - 1)*(1 - p)*(-alpha - beta)*np.log(x) + x**(-alpha - beta - 1)*(1 - p))*(alpha*p*x**(-alpha - 1)*np.log(x) - p*x**(-alpha - 1) - x**(-alpha - beta - 1)*(1 - p)*(-alpha - beta)*np.log(x) - x**(-alpha - beta - 1)*(1 - p))/(alpha*p*x**(-alpha - 1) - x**(-alpha - beta - 1)*(1 - p)*(-alpha - beta))**2
    )
    dll_dbeta2 = np.sum((-x**(-alpha - beta - 1)*(1 - p)*(-alpha - beta)*np.log(x)**2 - 2*x**(-alpha - beta - 1)*(1 - p)*np.log(x))/(alpha*p*x**(-alpha - 1) - x**(-alpha - beta - 1)*(1 - p)*(-alpha - beta)) + (-x**(-alpha - beta - 1)*(1 - p)
                                                                                                                                                                                                                 * (-alpha - beta)*np.log(x) - x**(-alpha - beta - 1)*(1 - p))*(x**(-alpha - beta - 1)*(1 - p)*(-alpha - beta)*np.log(x) + x**(-alpha - beta - 1)*(1 - p))/(alpha*p*x**(-alpha - 1) - x**(-alpha - beta - 1)*(1 - p)*(-alpha - beta))**2)
    dll_dp2 = np.sum((-alpha*x**(-alpha - 1) - x**(-alpha - beta - 1)*(-alpha - beta))*(alpha*x**(-alpha - 1) + x **
                                                                                       (-alpha - beta - 1)*(-alpha - beta))/(alpha*p*x**(-alpha - 1) - x**(-alpha - beta - 1)*(1 - p)*(-alpha - beta))**2)
    dll_dalphadbeta = np.sum((-x**(-alpha - beta - 1)*(1 - p)*(-alpha - beta)*np.log(x)**2 - 2*x**(-alpha - beta - 1)*(1 - p)*np.log(x))/(alpha*p*x**(-alpha - 1) - x**(-alpha - beta - 1)*(1 - p)*(-alpha - beta)) + (x**(-alpha - beta - 1)*(1 - p)*(-alpha - beta)*np.log(
        x) + x**(-alpha - beta - 1)*(1 - p))*(alpha*p*x**(-alpha - 1)*np.log(x) - p*x**(-alpha - 1) - x**(-alpha - beta - 1)*(1 - p)*(-alpha - beta)*np.log(x) - x**(-alpha - beta - 1)*(1 - p))/(alpha*p*x**(-alpha - 1) - x**(-alpha - beta - 1)*(1 - p)*(-alpha - beta))**2)
    dll_dalphadp = np.sum((alpha*x**(-alpha - 1) + x**(-alpha - beta - 1)*(-alpha - beta))*(alpha*p*x**(-alpha - 1)*np.log(x) - p*x**(-alpha - 1) - x**(-alpha - beta - 1)*(1 - p)*(-alpha - beta)*np.log(x) - x**(-alpha - beta - 1)*(1 - p))/(alpha*p*x**(-alpha - 1) -
                                                                                                                                                                                                                                               x**(-alpha - beta - 1)*(1 - p)*(-alpha - beta))**2 + (-alpha*x**(-alpha - 1)*np.log(x) + x**(-alpha - 1) - x**(-alpha - beta - 1)*(-alpha - beta)*np.log(x) - x**(-alpha - beta - 1))/(alpha*p*x**(-alpha - 1) - x**(-alpha - beta - 1)*(1 - p)*(-alpha - beta)))
    dll_dbetadp = np.sum((alpha*x**(-alpha - 1) + x**(-alpha - beta - 1)*(-alpha - beta))*(-x**(-alpha - beta - 1)*(1 - p)*(-alpha - beta)*np.log(x) - x**(-alpha - beta - 1)*(1 - p))/(alpha*p*x**(-alpha - 1) - x **
                                                                                                                                                                                       (-alpha - beta - 1)*(1 - p)*(-alpha - beta))**2 + (-x**(-alpha - beta - 1)*(-alpha - beta)*np.log(x) - x**(-alpha - beta - 1))/(alpha*p*x**(-alpha - 1) - x**(-alpha - beta - 1)*(1 - p)*(-alpha - beta)))
    return Hessian(dll_dalpha2, dll_dbeta2, dll_dp2, dll_dalphadbeta, dll_dalphadp, dll_dbetadp)
