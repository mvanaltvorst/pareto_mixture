from .type import Gradient, Hessian, ParetoMixtureParameters, Sample
import numpy as np

def fit_pareto_mixture_parameters(sample: Sample, k: int):
    subsample = sample.data
    subsample.sort()
    current_parameters = ParetoMixtureParameters(alpha=1.0, beta=1.0, p=0.5)

    subsample = Sample(subsample[:k])

    MAX_ITERS_WITHOUT_PROGRESS = 8
    MAX_ITERS = 100

    last_ll = float("inf")
    num_iters_without_progress = 0
    for _ in range(1, MAX_ITERS):
        ll, grad, hess = loglikelihood(current_parameters, subsample), gradient(current_parameters, subsample), hessian(current_parameters, subsample)
        if ll < last_ll + 0.0000001:
            num_iters_without_progress += 1
            if num_iters_without_progress >= MAX_ITERS_WITHOUT_PROGRESS:
                return current_parameters
        else:
            num_iters_without_progress = 0
        last_ll = ll

        dalpha, dbeta, dp = hess_inv_gradient(hess, grad)
        current_parameters.alpha -= dalpha
        current_parameters.beta -= dbeta
        current_parameters.p -= dp
    return "Did not converge in 100 iterations"

def move_to_feas_region(pm: ParetoMixtureParameters) -> ParetoMixtureParameters:
    if pm.p < 0.5:
        pm.p = 1 - pm.p
    if pm.p > 1.0:
        pm.p = 1.0
    if pm.beta < 1.0:
        pm.beta = 1.0
    if pm.p < 0.0:
        pm.p = 0.0
    if pm.p > 1.0:
        pm.p = 1.0
    return pm

def hess_inv_gradient(hess: Hessian, grad: Gradient):
    a11 = hess.dll_dp2 * hess.dll_dbeta2 - hess.dll_dbetadp * hess.dll_dbetadp
    a12 = hess.dll_dalphadp * hess.dll_dbetadp - hess.dll_dp2 * hess.dll_dalphadbeta
    a13 = hess.dll_dalphadbeta * hess.dll_dbetadp - hess.dll_dalphadp * hess.dll_dbeta2
    a22 = hess.dll_dp2 * hess.dll_dalpha2 - hess.dll_dalphadp * hess.dll_dalphadp
    a23 = hess.dll_dalphadbeta * hess.dll_dalphadp - hess.dll_dalpha2 * hess.dll_dbetadp
    a33 = hess.dll_dalpha2 * hess.dll_dbeta2 - hess.dll_dalphadbeta * hess.dll_dalphadbeta

    d = hess.dll_dalpha2 * a11 + hess.dll_dalphadbeta * a12 + hess.dll_dalphadp * a13

    return (
        (grad.dll_dalpha * a11 + grad.dll_dbeta * a12 + grad.dll_dp * a13) / d,
        (grad.dll_dalpha * a12 + grad.dll_dbeta * a22 + grad.dll_dp * a23) / d,
        (grad.dll_dalpha * a13 + grad.dll_dbeta * a23 + grad.dll_dp * a33) / d,
    )

def loglikelihood(pm: ParetoMixtureParameters, sample: Sample) -> float:
    alpha = pm.alpha
    beta = pm.beta
    p = pm.p
    x = sample.data
    return np.sum(
        np.log(p * x ** (-alpha - 1.) + (1 - p) * x ** (-alpha - beta - 1.))
    )

def gradient(pm: ParetoMixtureParameters, sample: Sample) -> Gradient:
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
