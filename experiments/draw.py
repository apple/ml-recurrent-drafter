import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D


def draw_2_sets_bar3d(
    x_l: List[int],
    y_l: List[int],
    height1_l: List[float],
    height2_l: List[float],
    x_label: str,
    y_label: str,
    z_label: str,
    pos: int,
    title: str,
    fig: plt.Figure,
) -> None:
    ax: Axes3D = fig.add_subplot(pos, projection="3d")

    # Create grid data with larger spacing between points
    x, y = np.array(x_l), np.array(y_l)
    z = np.zeros_like(x)  # Base of the bars
    height1 = np.array(height1_l)
    height2 = np.array(height2_l)  # noqa: F841
    # Width and depth of the bars
    dx = 0.3
    dy = 0.3 / 5

    # Offset the bars in the x-direction
    offset = 0.2  # Increased offset for larger space between bars

    # Plot the first set of bars
    ax.bar3d(x - offset, y - offset, z, dx, dy, height1, color="r", alpha=0.6)

    # Plot the second set of bars
    ax.bar3d(x + offset, y + offset, z, dx, dy, height2, color="#aaffaa", alpha=0.6)

    # Labels
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)

    ax.set_title(title)


def draw_from_log_wrt_beam_size(log_path: str, pos: int, title: str, fig: plt.Figure) -> None:
    with open(log_path, "r") as log:
        lines = log.readlines()
        width_list, length_list = [], []
        latency_list: List[float] = []
        forcasted_latency_list: List[float] = []
        base_latency = float(0)
        for i in range(len(lines) // 2):
            beam_line = lines[2 * i].split()
            width, length = int(beam_line[-3]), int(beam_line[-1])
            width_list.append(width)
            length_list.append(length)
            latency = float(lines[2 * i + 1].split()[-2])
            latency_list.append(latency)
            if width == 1 and length == 1:
                base_latency = latency
        assert base_latency > 0
        for width, length in zip(width_list, length_list):
            forcasted_latency_list.append(base_latency * width * length)

        draw_2_sets_bar3d(
            width_list,
            length_list,
            latency_list,
            forcasted_latency_list,
            "beam_width",
            "beam_length",
            "latency (ms)",
            pos=pos,
            title=title,
            fig=fig,
        )


LINEAR_PROJECTION_LOG = "mlx_w_50_l_10_linear_projection_m1_max.log"
SDPA_LOG = "mlx_w_50_l_10_sdpa_m1_max.log"

if __name__ == "__main__":
    # Enable interactive mode
    plt.ion()
    # Create figure and 3D axis
    fig = plt.figure(figsize=(6, 13))

    sdpa_log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), SDPA_LOG)
    draw_from_log_wrt_beam_size(
        sdpa_log_path, 211, title="MLX SDPA Latency w.r.t beam size", fig=fig
    )
    linear_projection_log_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), LINEAR_PROJECTION_LOG
    )
    draw_from_log_wrt_beam_size(
        linear_projection_log_path,
        212,
        title="MLX Linear Projection Latency w.r.t beam size",
        fig=fig,
    )
    # Show the plot
    plt.show()

    # Keep the plot open for interaction
    plt.ioff()
    plt.show()
