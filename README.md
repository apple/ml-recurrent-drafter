# Recurrent Drafter

This software project accompanies the research paper, [Recurrent Drafter for Fast Speculative Decoding in Large Language Models
](https://arxiv.org/abs/2403.09919).

## Installation

Run the following commands:

```shell
pip install --no-binary=protobuf -e ".[dev,train]"
```

The optional dependeny `dev` includes development tools like pre-commit, and `train` includes what the PyTorch-based training program needs.

Run unit tests in parallel:

```shell
pytest -n 8 --disable-warnings -s
```

## Train the Recurrent Drafter

A computer equipped with one or more contemporary CUDA GPUs. This training script, `train.py`, may use one or more GPUs via [Distributed Data Parallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html).

We use the [ShareGPT](https://huggingface.co/datasets/Aeala/ShareGPT_Vicuna_unfiltered) dataset to train the Vicuna 1.3 model's recurrent drafter, and we use the [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca_eval) dataset to evaluate the training results.

The training job fixes the parameters of the base Vicuna model and estimates only parameters of the drafter.  It takes about 1.5 hours to complete the training using the ShareGPT dataset and 8 H100 GPUs.

```
./recurrent_drafting/cmd/train.sh
```

## Run the Recurrent Drafter

The script `cmd/generate.py` supports both interactive mode and batch inference. Without the command-line option `--eval_mt_bench`, `generate.py` runs in interactive mode and waits for the user to type in the prompt via stdin. With the command-line option `--eval_mt_bench`, `generate.py` reads the input prompts from Hugging Face. To use the first, say 64, evaluation data instances, use: `--max_num_prompts=64`.

```
python3 -m recurrent_drafting.cmd.generate \
    --hf_tokenizer=lmsys/vicuna-7b-v1.3 \
    --hf_llm=lmsys/vicuna-7b-v1.3 \
    --hf_drafter=$HOME/m/redrafter \
    --eval_mt_bench=True \
    --max_prompt_length=500 \
    --max_generation_length=2048 \
    --beam_width=45 \
    --beam_length=5 \
    --greedy_search=True \
    --batch_size=8 \
    --dtype=bf16
```

To specify a certain GPU, say, the fourth one, use: `--use_gpu=3`. To run generate.py on CPU, use a negative GPU index: `--use_gpu=-1`

## Documentation

Please refer to [the documentation site](docs/index.md).

## Known Issues

For non-greedy decoding (temperature > 0), the current implementation might not give the exact distribution match and we are actively working to improve it. This does not impact greedy decoding when temperature = 0.

## Citation

If this project is useful for your work, please cite our paper:

```
@article{zhang2024recurrent,
  title={Recurrent Drafter for Fast Speculative Decoding in Large Language Models},
  author={Aonan Zhang and Chong Wang and Yi Wang and Xuanyu Zhang and Yunfei Cheng},
  journal={arXiv:2403.09919},
  year={2024},
  url={https://arxiv.org/abs/2403.09919},
  doi={10.48550/arXiv.2403.09919}
}
```
