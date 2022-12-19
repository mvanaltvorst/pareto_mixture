from dataclasses import dataclass
import numpy.typing as npt
import numpy as np

@dataclass
class ParetoMixtureParameters:
    alpha: float
    beta: float
    p: float


@dataclass
class Sample:
    data: npt.NDArray[np.float64]


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

