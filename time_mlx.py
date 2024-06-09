import time
from typing import Callable

import mlx.core as mx
import mlx.nn


def function(msg: str):
    """This decorator times the exeuction time of a function that calls MLX"""

    def decorator(g: Callable):
        def g_wrapped(*args, **kwargs):
            # Evaluate each of the input parameters to make sure they are ready before starting
            # ticking, and evaluate the return value(s) of g to make sure they are ready before
            # ending ticking.
            def eval_arg(arg):
                if (
                    isinstance(arg, mx.array)
                    or isinstance(arg, list)
                    or isinstance(arg, tuple)
                    or isinstance(arg, dict)
                ):
                    mx.eval(arg)
                elif isinstance(arg, mlx.nn.Module):
                    mx.eval(arg.parameters())
                return arg

            for arg in args:
                eval_arg(arg)
            for k, v in kwargs.items():
                eval_arg(v)
            tic = time.perf_counter()
            result = g(*args, **kwargs)
            eval_arg(result)
            timing = 1e3 * (time.perf_counter() - tic)
            print(f"{msg} {timing:.3f} (ms)")
            return result

        return g_wrapped

    return decorator
