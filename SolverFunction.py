import scipy
import numpy as np
import matplotlib.pyplot as plt
import symengine
from astropy import units as u
from astropy import constants as const
import pandas as pd
from astropy.units import Quantity
from astropy.visualization import quantity_support
from mpl_toolkits.mplot3d import Axes3D 
import mpl_toolkits.mplot3d as m3d
from math import lgamma, log, exp, comb
quantity_support()




def _find_unknown(kwargs):
    """Find the single argument set to None."""
    unknowns = [k for k, v in kwargs.items() if v is None]
    if len(unknowns) != 1:
        raise ValueError("Exactly one argument must be None")
    return unknowns[0]


def _strip_units(kwargs):
    """
    Separate numerical values and units.
    """
    values = {}
    units = {}

    for k, v in kwargs.items():
        if isinstance(v, Quantity):
            values[k] = v.value
            units[k] = v.unit
        else:
            values[k] = v
            units[k] = None

    return values, units

def auto_bracket(g, unit, scale=None, max_expand=20):
    """
    Automatically find a bracket [a, b] such that g(a) * g(b) < 0.
    g must take floats and return floats.
    """

    if scale is None:
        scale = 1.0

    x0 = scale
    f0 = g(x0)

    if f0 == 0:
        return x0, x0

    for n in range(max_expand):
        factor = 10**n

        a = -factor * x0
        b =  factor * x0

        fa = g(a)
        fb = g(b)

        if fa == 0:
            return a, a
        if fb == 0:
            return b, b

        if fa * fb < 0:
            return a, b

    raise RuntimeError(
        "Failed to bracket root automatically. "
        "Function may not cross zero."
    )


def solve_quantity(f, bracket=None, x0=None, **kwargs):
    """
    Solve f(...) = 0 for the argument set to None.
    Preserves astropy units.
    """

    # Identify unknown
    unk = _find_unknown(kwargs)

    # Infer unit of unknown
    unk_unit = None
    if isinstance(bracket, u.Quantity):
        unk_unit = bracket.unit
    elif isinstance(x0, u.Quantity):
        unk_unit = x0.unit
    else:
        # Try to infer from other arguments (rare, but possible)
        for v in kwargs.values():
            if isinstance(v, u.Quantity):
                unk_unit = v.unit
                break

    if unk_unit is None:
        raise ValueError(
            f"Cannot infer units for unknown '{unk}'. "
            "Provide bracket or x0 with units."
        )

    # Build scalar function for solver
    def g(x):
        local_kwargs = dict(kwargs)
        local_kwargs[unk] = x * unk_unit
        res = f(**local_kwargs)

        if not isinstance(res, u.Quantity):
            raise TypeError("Constraint must return an astropy Quantity")

        # Strip units *without* scaling
        return res.value

    # Solve
    if bracket is not None:
        if isinstance(bracket, u.Quantity):
            bracket = bracket.to(unk_unit).value
        a, b = bracket
        fa = g(a)
        fb = g(b)

        if fa == 0:
            return a * unk_unit
        if fb == 0:
            return b * unk_unit

        xtol = np.finfo(float).tiny
        rtol = 8 * np.finfo(float).eps

        sol = scipy.optimize.brentq(g, a, b, xtol=xtol, rtol=rtol, maxiter=200)
    else:
        if x0 is None:
            x0 = 1.0
        if isinstance(x0, u.Quantity):
            x0 = x0.to(unk_unit).value
        sol = scipy.optimize.fsolve(g, x0)[0]

    return sol * unk_unit

def simplify(quantity):
    return quantity.decompose()