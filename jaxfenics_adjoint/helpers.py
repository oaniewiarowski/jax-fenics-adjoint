import jax
from jax._src import ad_util
import jax.core as jax_core
from jax.core import ShapedArray
import numpy as np

import fecr
from fecr import from_numpy

import warnings

from typing import Type, List, Union, Iterable, Callable, Tuple

# Union[fenics.Constant, fenics.Function, firedrake.Constant, firedrake.Function, pyadjoint.AdjFloat]
BackendVariable = fecr._backends.BackendVariable
JAXArray = Union[jax.Array, jax.numpy.array, np.array]


def _abstract_to_numpy(aval: object) -> np.array:
    concrete_array = getattr(jax_core, "ConcreteArray", None)
    if concrete_array is not None and isinstance(aval, concrete_array):
        return aval.val

    if hasattr(aval, "shape"):
        dtype = getattr(aval, "dtype", None)
        return np.zeros(aval.shape, dtype=dtype)

    return np.asarray(aval)


def jax_to_fenics_numpy(jax_array: JAXArray, fem_variable: BackendVariable) -> np.array:
    """Convert JAX symbolic variables to concrete NumPy array compatible with FEniCS/Firedrake"""

    fem_backend = fecr._backends.get_backend(fem_variable)

    # JAX tracer specific part. Here we return zero values if tracer is not ConcreteArray type.
    if isinstance(jax_array, ad_util.Zero):
        if isinstance(fem_variable, fem_backend.lib.Constant):
            numpy_array = np.zeros_like(fem_variable.values())
            return numpy_array
        elif isinstance(fem_variable, fem_backend.lib.Function):
            numpy_array = np.zeros(fem_variable.vector().size())
            return numpy_array

    elif isinstance(jax_array, (jax_core.Tracer,)):
        aval = jax_core.get_aval(jax_array)
        return _abstract_to_numpy(aval)

    shape_dtype_struct = getattr(jax, "ShapeDtypeStruct", None)
    abstract_types = (ShapedArray,)
    if shape_dtype_struct is not None:
        abstract_types = abstract_types + (shape_dtype_struct,)

    if isinstance(jax_array, abstract_types):
        warnings.warn(
            "Got JAX tracer type to convert to FEniCS/Firedrake. Returning zero."
        )
        return _abstract_to_numpy(jax_array)

    else:
        numpy_array = np.asarray(jax_array)
        return numpy_array


def from_jax(
    jax_array: JAXArray, fem_variable: BackendVariable
) -> BackendVariable:  # noqa: C901
    """Convert numpy/jax array to FEniCS/Firedrake/pyadjoint variable"""
    return from_numpy(jax_to_fenics_numpy(jax_array, fem_variable), fem_variable)
