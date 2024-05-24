# Screen Recording

We use screen recording to visually present how fast an algorithm generates text or to compare the text generation speed of two algorithms.

![](./bf16-non-greedy.gif)

Inspired by [this tutorial](https://dev.to/noandrea/terminal-split-screen-recording-with-asciinema-screen-3b7i), we use [asciinema](https://asciinema.org/~noandrea) to record the terminal content. asciinema outputs a file that is much smaller than a video file.  We then use [agg](https://github.com/asciinema/agg) to convert the cast into a GIF file, or use [svg-term-cli](https://github.com/marionebl/svg-term-cli) for a SVG file.

When recording a comparative screen cast, we use [GNU screen](https://www.gnu.org/software/screen/) to split the terminal into two panels. And run the script generate.py simultaneously in each panel. We configure the split and run using the configuration file `~/.screenrc`, as follows:

```
split -v
screen 1 sh -c 'echo -e "Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions.\nexit" | TRANSFORMERS_VERBOSITY=error python3 -m recurrent_drafting.cmd.generate --hf_tokenizer=lmsys/vicuna-7b-v1.3 --hf_llm=lmsys/vicuna-7b-v1.3 --hf_drafter=$HOME/m/redrafter/ --max_prompt_length=1000 --max_generation_length=2048 --batch_size=1 --dtype=bf16 --greedy_search=False --temperature=0.01 --autoregression=True --use_gpu=1'
focus
screen 2 sh -c 'echo -e "Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions.\nexit" | TRANSFORMERS_VERBOSITY=error python3 -m recurrent_drafting.cmd.generate --hf_tokenizer=lmsys/vicuna-7b-v1.3 --hf_llm=lmsys/vicuna-7b-v1.3 --hf_drafter=$HOME/m/redrafter/ --max_prompt_length=1000 --max_generation_length=2048 --batch_size=1 --dtype=bf16 --greedy_search=False --temperature=0.01 --autoregression=False --beam_width=45 --beam_length=5 --use_gpu=2'
```

Please be aware that each run of the `generate.py` uses a unique GPU, so there must be at least two GPUs on the host.

We could now run the following command to run the programs and record the screen cast:

```shell
rm /tmp/a.cast; asciinema rec -c "screen -R devto" /tmp/a.cast
```

We could replay the cast:

```shell
asciinema play /tmp/a.cast
```

Or, we could convert it into a GIF file:

```shell
agg /tmp/a.cast /tmp/a.gif
```

## Install Dependencies

We could use the standard package manager to install asciinema. On Ubuntu, we use `apt`:

```shell
apt-get update
apt-get install asciinema screen
```

Agg has to be installed from the Rust source code.  To install Rust, run the following command:

```shell
curl https://sh.rustup.rs -sSf | sh
```

Then we need to update the shell and install agg

```shell
. ~/.bash_profile
cargo install --git https://github.com/asciinema/agg
```

To install `svg-term-cli`, we need npm:

```shell
apt-get install npm
```

Then, we could install `svg-term-cli`:

```shell
npm install -g svg-term-cli
```
