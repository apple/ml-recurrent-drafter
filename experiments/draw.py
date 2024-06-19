import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D


def draw_surface(
    x_l: List[int],
    y_l: List[int],
    z_l: List[float],
    x_label: str,
    y_label: str,
    z_label: str,
    pos: int,
    title: str,
    fig: plt.Figure,
) -> None:
    n_uniq_x = len(set(x_l))
    n_uniq_y = len(set(y_l))
    x = np.array(x_l, dtype=np.int32).reshape(n_uniq_x, n_uniq_y)
    y = np.array(y_l, dtype=np.int32).reshape(n_uniq_x, n_uniq_y)
    z = np.array(z_l, dtype=np.int32).reshape(n_uniq_x, n_uniq_y)

    ax: Axes3D = fig.add_subplot(pos, projection="3d")
    ax.plot_surface(x, y, z, cmap="autumn_r", rstride=1, cstride=1, alpha=0.7)
    ax.contour(x, y, z, 10, cmap="autumn_r", linestyles="solid", offset=-1)
    ax.contour(x, y, z, 10, colors="k", linestyles="solid")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    ax.set_title(title)


def parse_log_file(log_path: str) -> Tuple[List[int], List[int], List[float]]:
    beam_width_list: List[int] = []
    beam_length_list: List[int] = []
    latency_list: List[float] = []

    with open(log_path, "r") as log_file:
        for line_number, line in enumerate(log_file):
            if line_number % 2 == 0:
                fields = line.split()
                beam_width, beam_length = int(fields[-3]), int(fields[-1])
                beam_width_list.append(beam_width)
                beam_length_list.append(beam_length)
            else:
                latency = float(line.split()[-2])
                latency_list.append(latency)
    return beam_width_list, beam_length_list, latency_list


def draw_log_file(
    log_path: str, pos: int, title: str, enable_forcasted: bool, fig: plt.Figure
) -> None:
    draw_surface(
        *parse_log_file(log_path),
        "batch size",
        "sequence length",
        "latency (ms)",
        pos=pos,
        title=title,
        fig=fig,
    )


if __name__ == "__main__":
    plots: Dict[int, Tuple[str, str]] = {
        221: ("MLX SDPA", "result/mlx_w_125_l_25_sdpa_m1_max.log"),
        222: ("MLX Linear", "result/mlx_w_510_l_110_linear_projection_m1_max.log"),
        223: ("H100 SDPA", "result/torch_w_125_l_25_sdpa_h100.log"),
        224: ("H100 Linear", "result/torch_w_510_l_110_linear_projection_h100.log"),
    }

    fig = plt.figure(figsize=(12, 10))
    for subplot, info in plots.items():
        draw_log_file(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), info[1]),
            subplot,
            title=info[0],
            fig=fig,
            enable_forcasted=False,
        )
    plt.tight_layout()
    plt.savefig("/tmp/perf_contour.png")
