import math

import mlx
import mlx.core as mx

from . import time_mlx


@time_mlx.function("two_projs")
def two_projs():
    DIM = 1024

    @time_mlx.function("create_projs")
    def create_projs():
        p1 = mlx.nn.Linear(DIM, DIM, bias=False)
        p2 = mlx.nn.Linear(DIM, DIM, bias=False)
        return p1, p2

    @time_mlx.function("run_projs")
    def run_projs(p1, p2):
        x = mx.ones((1024, 1024))
        return p2(p1(x))

    p1, p2 = create_projs()
    run_projs(p1, p2)


def test_time_mlx_two_projs():
    for _ in range(10):
        ledger = time_mlx.ledger
        ledger.reset()
        two_projs()
        assert (
            ledger.records_dict["two_projscreate_projs"].timing[0]
            + ledger.records_dict["two_projsrun_projs"].timing[0]
            < ledger.records_dict["two_projs"].timing[0]
        )
        assert (
            math.fabs(
                ledger.records_dict["two_projscreate_projs"].timing[0]
                + ledger.records_dict["two_projsrun_projs"].timing[0]
                - ledger.records_dict["two_projs"].timing[0]
            )
            < 1  # ms
        )
