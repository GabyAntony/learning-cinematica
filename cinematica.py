import dataclasses
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


@dataclasses.dataclass(frozen=True)
class Ecuaciones:

    # ==========================================================
    # MRU
    # ==========================================================

    x0, x, v, t = sp.symbols("x0 x v t")

    MRU_x = sp.Eq(x, x0 + v*t)
    MRU_v = sp.Eq(v, (x - x0)/t)
    MRU_t = sp.Eq(t, (x - x0)/v)

    # ==========================================================
    # MRUA
    # ==========================================================

    v0, vf, a = sp.symbols("v0 vf a")

    MRUA_vf = sp.Eq(vf, v0 + a*t)

    MRUA_penx = sp.Eq(
        x - x0,
        v0*t + sp.Rational(1, 2)*a*t**2
    )

    MRUA_sin_t = sp.Eq(
        vf**2,
        v0**2 + 2*a*(x - x0)
    )

    MRUA_a = sp.Eq(
        a,
        (vf - v0)/t
    )

    MRUA_vmedia = sp.Eq(
        sp.Symbol("vmedia"),
        (v0 + vf)/2
    )

    # ==========================================================
    # MOVIMIENTO PARABÓLICO
    # ==========================================================

    v0x, v0y = sp.symbols("v0x v0y")
    theta, g, y0 = sp.symbols("theta g y0")

    PAR_v0x = sp.Eq(
        v0x,
        v0 * sp.cos(theta)
    )

    PAR_v0y = sp.Eq(
        v0y,
        v0 * sp.sin(theta)
    )

    PAR_x = sp.Eq(
        x,
        x0 + v0x*t
    )

    PAR_y = sp.Eq(
        sp.Symbol("y"),
        y0 + v0y*t - sp.Rational(1, 2)*g*t**2
    )

    PAR_vy = sp.Eq(
        sp.Symbol("vy"),
        v0y - g*t
    )

    PAR_tmax = sp.Eq(
        sp.Symbol("tmax"),
        v0y/g
    )

    PAR_hmax = sp.Eq(
        sp.Symbol("hmax"),
        y0 + v0y**2/(2*g)
    )

    # ==========================================================
    # MOVIMIENTO CIRCULAR
    # ==========================================================

    r, omega, T, f = sp.symbols("r omega T f")
    vc, ac = sp.symbols("vc ac")

    CIRC_v = sp.Eq(
        vc,
        omega*r
    )

    CIRC_T = sp.Eq(
        T,
        2*sp.pi/omega
    )

    CIRC_f = sp.Eq(
        f,
        1/T
    )

    CIRC_omega = sp.Eq(
        omega,
        2*sp.pi*f
    )

    CIRC_ac = sp.Eq(
        ac,
        vc**2/r
    )

    CIRC_ac_omega = sp.Eq(
        ac,
        omega**2*r
    )  

@dataclasses.dataclass
class Ejercicio:
    tema: str
    tipo: str
    datos: dict
    variable: sp.Symbol
    respuesta: float
    tolerancia: float = 0.01

class calculator():
    def __init__(self) -> None:
        pass
   
class simulator():
    def __init__(self) -> None:
        pass

class challenge():
    def __init__(self) -> None:
        pass

def main():
    pass
