from dataclasses import dataclass
import numpy.typing as npt
import numpy as np

@dataclass
class ParetoMixtureParameters:
    alpha: float
    beta: float
    p: float

    def __add__(self, other):
        return ParetoMixtureParameters(
            self.alpha + other.dalpha,
            self.beta + other.dbeta,
            self.p + other.dp,
        )

@dataclass
class DeltaParetoMixtureParameters:
    dalpha: float
    dbeta: float
    dp: float
    # scalar multiply
    def __mul__(self, other):
        return DeltaParetoMixtureParameters(
            self.dalpha * other,
            self.dbeta * other,
            self.dp * other,
        )

    # add to ParetoMixtureParameters
    def __add__(self, other):
        return ParetoMixtureParameters(
            self.dalpha + other.alpha,
            self.dbeta + other.beta,
            self.dp + other.p,
        )


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

