#!/usr/bin/env bash

source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate mlx_recurrent_drafting

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 $script_dir/benchmark_llama.py
