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


def average_autoregression_time(autoregression_csv: str) -> Tuple[float, float]:
    """benchmark_autoregression.py prints the comprehension and generation time."""
    df = pandas.read_csv(autoregression_csv)
    df = df[df["dtype"] == "mlx.core.float16"]
    return float(df["comprehension_ours"].mean()), float(df["generation_ours"].mean())


(average_comprehension_time, average_autoregression_generation_time) = average_autoregression_time(
    "/tmp/autoregression.csv"
)


def derive_perf_versus_beam_shape(
    recurrent_drafting_csv: str,
    average_comprehension_time: float,
    average_autoregression_generation_time: float,
) -> pandas.DataFrame:
    df = pandas.read_csv(recurrent_drafting_csv)
    df["generated_length"] = df["prompt_and_generated_length"] - df["prompt_length"]
    df["generation_time"] = df["comprehension_and_generation_time"] - average_comprehension_time
    df["tokens_per_sec"] = df["generated_length"] / df["generation_time"] * 1000.0
    df["speedup"] = average_autoregression_generation_time / df["generation_time"]
    return df


df = derive_perf_versus_beam_shape(
    "/tmp/recurrent_drafting.csv",
    average_comprehension_time,
    average_autoregression_generation_time,
)


def plot_groups(df: pandas.DataFrame) -> None:
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
    for i, (group_name, group_df) in enumerate(groups.items(), 1):
        max_row = group_df.loc[group_df["speedup"].idxmax()]
        max_label = (
            f"beam shape=({max_row['beam_width']},{max_row['beam_length']}) "
            + f"{max_row['tokens_per_sec']:.3f} tokens/sec speedup={max_row['speedup']:.3f}"
        )

        ax: Axes3D = fig.add_subplot(2, 2, i, projection="3d")
        ax.plot_trisurf(
            group_df["beam_width"], group_df["beam_length"], group_df["speedup"], cmap="viridis"
        )
        ax.set_xlabel("beam width")
        ax.set_ylabel("beam length")
        ax.set_zlabel("speedup")
        ax.set_title(group_name + "\n" + max_label)
        # plt.tight_layout()
        # fig.suptitle("Speedup of Recurrent Drafting over Autoregression on M1 Max", fontsize=16)
        plt.savefig("/tmp/p.pdf")


plot_groups(df)
