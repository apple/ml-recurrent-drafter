import time
from typing import Callable, Dict, List

import mlx.core as mx
import mlx.nn
from tabulate import tabulate


class _Record:
    def __init__(self, msg: str, indentation: int) -> None:
        self.msg = msg
        self.indentation = indentation
        self.timing: List[float] = []


class Ledger:
    def __init__(self) -> None:
        self.records: List[_Record] = []
        self.records_dict: Dict[str, _Record] = {}
        self.indentation = -1
        self.key = ""

    def reset(self):
        self.records = []
        self.records_dict: Dict[str, _Record] = {}
        self.indentation = -1
        self.key = ""

    def print_table(self):
        table = [
            ["-" * r.indentation + "> " + r.msg, sum(r.timing) / len(r.timing), sum(r.timing)]
            for r in self.records
        ]
        print(
            tabulate(
                table,
                headers=["function", "latency per run (ms)", "latency in total (ms)"],
                tablefmt="psql",
            )
        )

    def print_summary(self):
        for r in self.records:
            print(f"{r.msg} {sum(r.timing):.3f} (ms)")


ledger = Ledger()


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

            ledger.indentation += 1
            prev_key = ledger.key

            ledger.key += msg
            if ledger.key not in ledger.records_dict:
                r = _Record(msg, ledger.indentation)
                ledger.records.append(r)
                ledger.records_dict[ledger.key] = r

            tic = time.perf_counter()
            result = g(*args, **kwargs)
            eval_arg(result)
            timing = 1e3 * (time.perf_counter() - tic)
            ledger.records_dict[ledger.key].timing.append(timing)

            ledger.indentation -= 1
            ledger.key = prev_key

            return result

        return g_wrapped

    return decorator
