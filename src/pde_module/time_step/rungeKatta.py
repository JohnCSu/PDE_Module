"""Runge-Kutta time integration schemes for PDE solves."""

import warp as wp
from ..stencil.stencil import Stencil
from warp.types import (
    vector,
    matrix,
    type_is_matrix,
    type_is_vector,
    types_equal,
    is_array,
)
from ..stencil.hooks import *
from .forwardEuler import ForwardEuler
from ..utils import tuplify, SignatureMismatchError
from pde_module.utils.types import wp_Array,wp_Vector,wp_Matrix
from typing import Callable
import inspect


class RungeKatta1(Stencil):
    """Runge-Kutta 1 (Forward Euler) for multiple fields.

    Simple first-order time integration using Forward Euler.
    """

    def __init__(
        self,
        field_dtypes: wp_Vector | wp_Matrix | tuple[wp_Vector | wp_Matrix, ...],
        f: Callable,
        bc: Callable,
    ) -> None:
        """Initialize RK1 integrator.

        Args:
            field_dtypes: Dtype(s) of fields to integrate.
            f: Function computing time derivatives: f(t, *fields, **kwargs).
            bc: Function applying boundary conditions: bc(t, *fields, **kwargs).
        """
        super().__init__()
        self.input_dtypes = tuplify(field_dtypes)
        self.forwardEulers = tuple(
            ForwardEuler(input_dtype) for input_dtype in self.input_dtypes
        )
        self.f = f
        self.bc = bc
        self.num_fields = len(self.input_dtypes)

        if inspect.signature(f) != inspect.signature(bc):
            raise SignatureMismatchError(
                f"Signature For f {inspect.signature(f)} does not match the signature of bc {inspect.signature(bc)}!"
            )

    @setup
    def check_args(self, t: float, dt: float, *arrays: wp.array, **kwargs) -> None:
        """Verify that array dtypes match the expected input dtypes."""
        for arr, time_step in zip(arrays, self.forwardEulers):
            assert is_array(arr)
            assert types_equal(arr.dtype, time_step.input_dtype), (
                "arrays and input dtype for euler must match"
            )

    def forward(
        self, t: float, dt: float, *arrays: wp.array, **kwargs
    ) -> tuple[wp_Array, ...]:
        """Perform one RK1 time step.

        Args:
            t: Current time.
            dt: Time step size.
            *arrays: Field arrays to advance.
            **kwargs: Additional arguments passed to f and bc.

        Returns:
            Tuple of advanced field arrays.
        """
        arrays = self.bc(t, *arrays, **kwargs)
        k1s = self.f(t, *arrays, **kwargs)
        return tuple(
            forwardEuler(arr, k1, dt / 2)
            for forwardEuler, arr, k1 in zip(self.forwardEulers, arrays, k1s)
        )


class RungeKatta2(Stencil):
    """Runge-Kutta 2 (Midpoint Method) for multiple fields.

    Implements:
        k1 = f(t_n, y_n)
        k2 = f(t_n + dt/2, y_n + dt/2 * k1)
        y_{n+1} = y_n + dt * k2
    """

    def __init__(
        self,
        field_dtypes: wp_Vector | wp_Matrix | tuple[wp_Vector | wp_Matrix, ...],
        f: Callable,
        bc: Callable,
    ) -> None:
        """Initialize RK2 integrator.

        Args:
            field_dtypes: Dtype(s) of fields to integrate.
            f: Function computing time derivatives: f(t, *fields, **kwargs).
            bc: Function applying boundary conditions: bc(t, *fields, **kwargs).

        Raises:
            SignatureMismatchError: If f and bc have different signatures.
        """
        super().__init__()
        self.input_dtypes = tuplify(field_dtypes)
        self.forwardEulers_1 = tuple(
            ForwardEuler(input_dtype) for input_dtype in self.input_dtypes
        )
        self.forwardEulers_2 = tuple(
            ForwardEuler(input_dtype) for input_dtype in self.input_dtypes
        )
        self.f = f
        self.bc = bc
        self.num_fields = len(self.input_dtypes)

        if inspect.signature(f) != inspect.signature(bc):
            raise SignatureMismatchError(
                f"Signature For f {inspect.signature(f)} does not match the signature of bc {inspect.signature(bc)}!"
            )

    @setup
    def check_args(self, t: float, dt: float, *arrays: wp.array, **kwargs) -> None:
        """Verify that array dtypes match the expected input dtypes."""
        for arr, time_step in zip(arrays, self.forwardEulers_1):
            assert is_array(arr)
            assert types_equal(arr.dtype, time_step.input_dtype), (
                "arrays and input dtype for euler must match"
            )

    @setup
    def set_buffers(self, t: float, dt: float, *arrays: wp.array, **kwargs) -> None:
        """Allocate buffer arrays for intermediate values."""
        self.t0_fields = tuple(wp.empty_like(arr) for arr in arrays)

    def _copy_to_buffers(self, arrays: tuple[wp.array, ...]) -> None:
        """Copy arrays to buffer storage."""
        for i, arr in enumerate(arrays):
            wp.copy(self.t0_fields[i], arr)

    def forward(
        self, t: float, dt: float, *arrays: wp.array, **kwargs
    ) -> tuple[wp_Array, ...]:
        """Perform one RK2 time step.

        Args:
            t: Current time.
            dt: Time step size.
            *arrays: Field arrays to advance.
            **kwargs: Additional arguments passed to f and bc.

        Returns:
            Tuple of advanced field arrays.
        """
        arrays = self.bc(t, *arrays, **kwargs)
        self._copy_to_buffers(arrays)
        k1s = self.f(t, *arrays, **kwargs)
        y1s = [
            forwardEuler(arr, k1, dt / 2)
            for forwardEuler, arr, k1 in zip(self.forwardEulers_1, arrays, k1s)
        ]

        y1s = self.bc(t, *y1s, **kwargs)
        k2s = self.f(t + dt / 2, *y1s, **kwargs)

        return tuple(
            forwardEuler(arr, k2, dt)
            for forwardEuler, arr, k2 in zip(self.forwardEulers_2, self.t0_fields, k2s)
        )
