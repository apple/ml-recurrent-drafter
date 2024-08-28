# Copyright © 2024 Apple Inc.
from typing import List

import absl.app
import absl.flags
import matplotlib.animation
import matplotlib.pyplot
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

FLAGS = absl.flags.FLAGS


def main(_) -> None:
    data = np.loadtxt(FLAGS.csv_file, delimiter=",", skiprows=1)
    x = data[:, 0]
    y = data[:, 1]
    z = [data[:, 2], data[:, 3]]

    max_num_beam = len(set(x))
    max_beam_len = len(set(y))

    X = np.array(x).reshape(max_num_beam, max_beam_len)
    Y = np.array(y).reshape(X.shape)
    Z = [np.array(ζ).reshape(X.shape) for ζ in z]

    fig = matplotlib.pyplot.figure(figsize=(12, 6))

    axes: List[Axes3D] = [
        fig.add_subplot(121, projection="3d"),
        fig.add_subplot(122, projection="3d"),
    ]
    [ax.set_proj_type("persp") for ax in axes]
    [ax.set_xlabel("beam width") for ax in axes]
    [ax.set_ylabel("beam length") for ax in axes]
    [axes[i].plot_surface(X, Y, Z[i], cmap="viridis") for i in range(2)]
    [
        axes[i].set_zlabel(zlabel)
        for i, zlabel in enumerate(
            ["Average number of accepted tokens per step", "Tokens per second"]
        )
    ]

    def animate(frame):
        [ax.view_init(elev=15, azim=frame) for ax in axes]
        return (fig,)

    ani = matplotlib.animation.FuncAnimation(fig, animate, frames=np.arange(0, 360, 2), interval=50)
    writer = matplotlib.animation.FFMpegWriter()
    ani.save(FLAGS.animation_file, writer=writer)


if __name__ == "__main__":
    absl.flags.DEFINE_string(
        "csv_file",
        default=None,
        help="The CSV file with columns beam_width, beam_length, ANAT, tokens/sec from"
        + "awk -f recurrent_drafting/benchmark/perf_versus_candidates_len_and_num/make_csv.awk "
        + " nohup \nwhere nohup is the log file from perf_versus_candidates_len_and_num.bash",
        required=True,
    )
    absl.flags.DEFINE_string(
        "animation_file",
        default="/tmp/animation.mov",
        help="The output ffmpeg gif file.",
    )
    absl.app.run(main)
