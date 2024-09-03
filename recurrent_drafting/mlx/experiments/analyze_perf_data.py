#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
"""Usage:
1. Run benchmark_autoregression.py to genreate /tmp/autoregression.csv
2. Run benchmark_recurrent_drafting.py to generate /tmp/recurrent_drafting.csv
3. Run this script to generate /tmp/p.pdf
"""
from typing import Tuple

import matplotlib.pyplot as plt
import pandas
from mpl_toolkits.mplot3d import Axes3D


def average_autoregression_time(autoregression_csv: str) -> Tuple[float, ...]:
    """benchmark_autoregression.py prints the comprehension and generation time."""
    df = pandas.read_csv(autoregression_csv)
    fp16 = df[df["dtype"] == "mlx.core.float16"]
    bf16 = df[df["dtype"] == "mlx.core.bfloat16"]
    return (
        float(fp16["comprehension_ours"].mean()),
        float(fp16["generation_ours"].mean()),
        float(bf16["comprehension_ours"].mean()),
        float(bf16["generation_ours"].mean()),
    )


(
    average_comprehension_time_fp16,
    average_autoregression_generation_time_fp16,
    average_comprehension_time_bf16,
    average_autoregression_generation_time_bf16,
) = average_autoregression_time("/tmp/autoregression.csv")


def plot_groups(
    recurrent_drafting_csv: str,
    average_comprehension_time_fp16,
    average_autoregression_generation_time_fp16,
    average_comprehension_time_bf16,
    average_autoregression_generation_time_bf16,
) -> None:
    df = pandas.read_csv(recurrent_drafting_csv)
    df = df[df["run"] == 0]

    groups = {
        "float16 greedy": df[
            (df["dtype"] == "mlx.core.float16") & (df["greedy"] == True)  # noqa: E712
        ],
        "float16 non-greedy": df[
            (df["dtype"] == "mlx.core.float16") & (df["greedy"] == False)  # noqa: E712
        ],
        "bfloat16 greedy": df[
            (df["dtype"] == "mlx.core.bfloat16") & (df["greedy"] == True)  # noqa: E712
        ],
        "bfloat16 non-greedy": df[
            (df["dtype"] == "mlx.core.bfloat16") & (df["greedy"] == False)  # noqa: E712
        ],
    }

    # Create a figure with 2x2 subplots
    fig = plt.figure(figsize=(14, 10))

    # Iterate over each group and plot the 3D terrain using the new Z axis
    for i, (grp_name, grp) in enumerate(groups.items(), 1):
        grp["generated_length"] = grp["prompt_and_generated_length"] - grp["prompt_length"]
        grp["generation_time"] = grp["comprehension_and_generation_time"] - (
            average_comprehension_time_bf16
            if "bfloat16" in grp_name
            else average_comprehension_time_fp16
        )
        grp["tokens_per_sec"] = grp["generated_length"] / grp["generation_time"] * 1000.0
        grp["speedup"] = (
            average_autoregression_generation_time_bf16
            if "bfloat16" in grp_name
            else average_autoregression_generation_time_fp16
        ) / grp["generation_time"]
        print(grp)  # debug

        max_row = grp.loc[grp["speedup"].idxmax()]
        print(max_row)  # debug
        max_label = (
            f"beam shape=({max_row['beam_width']},{max_row['beam_length']}) "
            + f"{max_row['tokens_per_sec']:.3f} tokens/sec speedup={max_row['speedup']:.3f}"
        )

        ax: Axes3D = fig.add_subplot(2, 2, i, projection="3d")
        ax.plot_trisurf(grp["beam_width"], grp["beam_length"], grp["speedup"], cmap="viridis")
        ax.set_xlabel("beam width")
        ax.set_ylabel("beam length")
        ax.set_zlabel("speedup")
        ax.set_title(grp_name + "\n" + max_label)
        # plt.tight_layout()
        # fig.suptitle("Speedup of Recurrent Drafting over Autoregression on M1 Max", fontsize=16)
    plt.savefig("/tmp/p.pdf")
    plt.show()


plot_groups(
    "/tmp/recurrent_drafting.csv",
    average_comprehension_time_fp16,
    average_autoregression_generation_time_fp16,
    average_comprehension_time_bf16,
    average_autoregression_generation_time_bf16,
)
