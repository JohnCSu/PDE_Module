"""Hooks to register methods on stencil objects."""

from typing import Callable, Optional


def setup(
    fn: Optional[Callable] = None, *, order: int = 0, debug: bool = False
) -> Callable:
    """Hook to register a method to run during setup.

    Args:
        fn: Function to wrap. With decorator notation this fn can be ignored.
        order: Order of hooked function. Lower value (negative allowed) means
            the function is run earlier.
        debug: If True, the wrapped function is registered only if the debug flag
            in the stencil is set to True. Useful for debugging or for setting safer
            but less performant code (such as array and dtype checking).

    Returns:
        Decorated function with hook metadata.
    """
    assert isinstance(order, int), "order must be integer"

    def decorator(func: Callable) -> Callable:
        assert all(
            hasattr(func, attr) is False
            for attr in ["_setup_order", "_after_forward", "_before_forward"]
        ), "Method already registered!"
        func._setup_order = order
        func._debug = debug
        return func

    if fn is None:
        return decorator
    else:
        return decorator(fn)


def before_forward(
    fn: Optional[Callable] = None, *, order: int = 0, debug: bool = False
) -> Callable:
    """Hook to register a method to run before forward pass.

    Args:
        fn: Function to wrap. With decorator notation this fn can be ignored.
        order: Order of hooked function. Lower value (negative allowed) means
            the function is run earlier.
        debug: If True, the wrapped function is registered only if the debug flag
            in the stencil is set to True. Useful for debugging or for setting safer
            but less performant code.

    Returns:
        Decorated function with hook metadata.
    """
    assert isinstance(order, int), "order must be integer"

    def decorator(func: Callable) -> Callable:
        assert all(
            hasattr(func, attr) is False
            for attr in ["_setup_order", "_after_forward", "_before_forward"]
        ), "Method already registered!"
        func._before_forward_order = order
        func._debug = debug
        return func

    if fn is None:
        return decorator
    else:
        return decorator(fn)


def after_forward(
    fn: Optional[Callable] = None, *, order: int = 0, debug: bool = False
) -> Callable:
    """Hook to register a method to run after forward pass.

    Args:
        fn: Function to wrap. With decorator notation this fn can be ignored.
        order: Order of hooked function. Lower value (negative allowed) means
            the function is run earlier.
        debug: If True, the wrapped function is registered only if the debug flag
            in the stencil is set to True. Useful for debugging or for setting safer
            but less performant code.

    Returns:
        Decorated function with hook metadata.
    """
    assert isinstance(order, int), "order must be integer"

    def decorator(func: Callable) -> Callable:
        assert all(
            hasattr(func, attr) is False
            for attr in ["_setup_order", "_after_forward", "_before_forward"]
        ), "Method already registered!"
        func._after_forward_order = order
        func._debug = debug
        return func

    if fn is None:
        return decorator
    else:
        return decorator(fn)
