from dataclasses import dataclass
import numpy as np
from typing import Tuple, Union
from dataclasses import dataclass

@dataclass
class ParetoMixtureParameters:
    alpha: float
    beta: float
    p: float

def fit_pareto_mixture_parameters(subsample: list, k: int) -> Union[ParetoMixtureParameters, str]:
    subsample.sort()
    current_parameters = ParetoMixtureParameters(super.hill(subsample, k), 1., 0.8)
    subsample = subsample[:k]

    MAX_ITERS_WITHOUT_PROGRESS = 8
    MAX_ITERS = 100

    last_ll = float('inf')
    num_iters_without_progress = 0
    for _ in range(MAX_ITERS):
        ll, grad, hess = calculate_ll_gradient_hessian(subsample, current_parameters)
        if ll < last_ll + 1e-7:
            num_iters_without_progress += 1
            if num_iters_without_progress >= MAX_ITERS_WITHOUT_PROGRESS:
                return current_parameters
        else:
            num_iters_without_progress = 0
        last_ll = ll

        dalpha, dbeta, dp = hess_inv_gradient(hess, grad)
        current_parameters = ParetoMixtureParameters(current_parameters.alpha - dalpha, current_parameters.beta - dbeta, current_parameters.p - dp)
    return "Did not converge in 100 iterations"

def hess_inv_gradient(hess: Hessian, grad: Gradient) -> Tuple[float, float, float]:
    a11 = hess.dll_dbeta2 * hess.dll_dp2 - hess.dll_dbetadp * hess.dll_dbetadp
    a12 = hess.dll_dalphadp * hess.dll_dbetadp - hess.dll_dp2 * hess.dll_dalphadbeta
    a13 = hess.dll_dalphadbeta * hess.dll_dbetadp - hess.dll_dalphadp * hess.dll_dbeta2
    a22 = hess.dll_dp2 * hess.dll_dalpha2 - hess.dll_dalphadp * hess.dll_dalphadp
    a23 = hess.dll_dalphadbeta * hess.dll_dalphadp - hess.dll_dalpha2 * hess.dll_dbetadp
    a33 = hess.dll_dalpha2 * hess.dll_dbeta2 - hess.


@dataclass
class ParetoMixture:
    alpha: float
    beta: float
    p: float


@dataclass
class Sample:
    data: list[float]


@dataclass
class Gradient:
    dll_dalpha: float
    dll_dbeta: float
    dll_dp: float


@dataclass
class Hessian:
    dll_dalpha2: float
    dll_dbeta2: float
    dll_dp2: float
    dll_dalphadbeta: float
    dll_dalphadp: float
    dll_dbetadp: float


def gradient(pm: ParetoMixture, sample: Sample) -> Gradient:
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


def hessian(pm: ParetoMixture, sample: Sample) -> Hessian:
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


def fit():
    pass
