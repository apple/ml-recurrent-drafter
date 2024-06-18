import os
from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D


@dataclass
class Heights:
    heights: List[float]
    color: str


def draw_sets_bar3d(
    x_l: List[int],
    y_l: List[int],
    heights_list: List[Heights],
    x_label: str,
    y_label: str,
    z_label: str,
    pos: int,
    title: str,
    fig: plt.Figure,
) -> None:
    ax: Axes3D = fig.add_subplot(pos, projection="3d")

    # Create grid data with larger spacing between points
    x, y = np.array(x_l, dtype=np.int32), np.array(y_l, dtype=np.int32)
    z = np.zeros_like(x)  # Base of the bars
    # Width and depth of the bars
    dx = (max(x_l) - min(x_l)) / len(set(x_l)) / 2
    dy = (max(y_l) - min(y_l)) / len(set(y_l)) / 2

    # Offset the bars in the x-direction
    offset = 0.2  # Increased offset for larger space between bars

    # Plot the bars
    for h in heights_list:
        ax.bar3d(x - offset, y - offset, z, dx, dy, np.array(h.heights), color=h.color, alpha=0.6)

    # Labels
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)

    ax.set_title(title)


def draw_from_log_wrt_beam_size(
    log_path: str, pos: int, title: str, enable_forcasted: bool, fig: plt.Figure
) -> None:
    min_width, min_length = 0, 0
    with open(log_path, "r") as log:
        lines = log.readlines()
        width_list, length_list = [], []
        latency_list: List[float] = []
        base_latency = float(0)
        for i in range(len(lines) // 2):
            beam_line = lines[2 * i].split()
            width, length = int(beam_line[-3]), int(beam_line[-1])
            width_list.append(width)
            length_list.append(length)
            latency = float(lines[2 * i + 1].split()[-2])
            latency_list.append(latency)
            if (min_width == 0 and min_length == 0) or (width < min_width and length < min_length):
                base_latency = latency
        assert base_latency > 0
        forcasted_latency_list = [
            base_latency * width * length for width, length in zip(width_list, length_list)
        ]

        draw_sets_bar3d(
            width_list,
            length_list,
            (
                [Heights(latency_list, "r"), Heights(forcasted_latency_list, "#aaffaa")]
                if enable_forcasted
                else [Heights(latency_list, "r")]
            ),
            "beam_width",
            "beam_length",
            "latency (ms)",
            pos=pos,
            title=title,
            fig=fig,
        )


LINEAR_PROJECTION_LOG_M1_MAX = "result/mlx_w_510_l_110_linear_projection_m1_max.log"
SDPA_LOG_M1_MAX = "result/mlx_w_125_l_25_sdpa_m1_max.log"
LINEAR_PROJECTION_LOG_H100 = "result/torch_w_510_l_110_linear_projection_h100.log"
SDPA_LOG_H100 = "result/torch_w_125_l_25_sdpa_h100.log"

if __name__ == "__main__":
    # Enable interactive mode
    plt.ion()
    # Create figure and 3D axis
    fig = plt.figure(figsize=(12, 10))

    draw_from_log_wrt_beam_size(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), SDPA_LOG_M1_MAX),
        221,
        title="MLX SDPA Latency w.r.t beam size",
        fig=fig,
        enable_forcasted=False,
    )
    draw_from_log_wrt_beam_size(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), LINEAR_PROJECTION_LOG_M1_MAX),
        222,
        title="MLX Linear Projection Latency w.r.t beam size",
        fig=fig,
        enable_forcasted=False,
    )
    draw_from_log_wrt_beam_size(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), SDPA_LOG_H100),
        223,
        title="H100 SDPA Latency w.r.t beam size",
        fig=fig,
        enable_forcasted=False,
    )
    draw_from_log_wrt_beam_size(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), LINEAR_PROJECTION_LOG_H100),
        224,
        title="H100 Linear Projection Latency w.r.t beam size",
        fig=fig,
        enable_forcasted=False,
    )
    # Show the plot
    plt.show()

    # Keep the plot open for interaction
    plt.ioff()
    plt.show()
