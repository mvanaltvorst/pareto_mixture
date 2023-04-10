from .type import Gradient, Hessian, ParetoMixtureParameters, Sample
import numpy as np
from collections import defaultdict
import pandas as pd
from numpy.linalg import inv
from math import pow


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
        np.log(p * x ** (-alpha) + (1 - p) * x ** (-beta))
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
        -p*(x**(-alpha))*np.log(x)/(p*(x**(-alpha)) + (x**(-beta))*(1 - p))
    )
    dll_dbeta = np.sum(
        -(x**(-beta))*(1 - p)*np.log(x)/(p*(x**(-alpha)) + (x**(-beta))*(1 - p))
    )
    dll_dp = np.sum(
        (-(x**(-beta)) + (x**(-alpha)))/(p*(x**(-alpha)) + (x**(-beta))*(1 - p))
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
