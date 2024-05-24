#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
"""This module records stats in each call to recurrent_drafting.generate. Each call takes a batch of
prompts and runs a text generation loop. Each loop step appends one or more tokens at the end
of each sequence in the batch. By invoking functions in this module in the text generation
code as the following, we records the timing and non-timing stats and summarize them into
human readable form.

def generate(self: RecurrentDrafting, batch: torch.Tensor, r: stats.TextGeneration):
    for _ in range(whatever):
        with r.step() as step_record:
            with step_record.time("a_task"):
                self.a_task(batch)
            step_record.set(input_id_length, avg_num_accepted)

ledger = stats.Ledger()
for batch in batches:
    with recurrent_drafting.states.text_generation(ledger) as r:  # Fill in ledger.
        generate(redrafter_model, batch, r)

stats = recurrent_drafting.stats.summarize(ledger)  # Scan ledger.
recurrent_drafting.stats.draw_table("title", stats)
recurrent_drafting.stats.draw_figure("title", stats, "/tmp/a.png")
del ledger
"""

from __future__ import (
    annotations,  # No need to refer to not-yet-defined type names as strings
)

import contextlib
import dataclasses
import os
import tempfile
import time
from collections import defaultdict
from typing import Any, DefaultDict, Dict, Generator, List, Tuple

import jsonlines
import numpy as np
import torch
from tabulate import SEPARATING_LINE, tabulate  # type: ignore


class TextGeneration:
    """Each TextGeneration holds stats from a call to RecurrentDrafting.generate."""

    def __init__(self: TextGeneration) -> None:
        self.n_steps: int = 0  # The number of text generation loop steps executed so far.
        self.wall_times: Dict[str, List[float]] = {}  # Function name to seconds per step.
        self.stats: Dict[str, Any] = {
            "step": [],  # A zero-based array incremented by one.
            "wall_times": self.wall_times,
        }

    @contextlib.contextmanager
    def step(self: TextGeneration) -> Generator[Step, None, None]:
        r = TextGeneration.Step(self)
        self.stats["step"].append(self.n_steps)
        self.n_steps += 1
        yield r

    class Step:
        """TextGeneration.Step appends timing and non-timing stats to TextGeneration."""

        def __init__(self: TextGeneration.Step, gr: TextGeneration) -> None:
            self._gr = gr

        @contextlib.contextmanager
        def time(self: TextGeneration.Step, key: str) -> Generator[None, None, None]:
            """Append a wall-time to a wall_times field in TextGeneration."""
            with torch.profiler.record_function(key):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()
                yield
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end = time.perf_counter()
                elapsed_time = end - start
                if key in self._gr.wall_times:
                    assert len(self._gr.wall_times[key]) == self._gr.n_steps - 1
                    self._gr.wall_times[key].append(elapsed_time)
                else:
                    self._gr.wall_times[key] = [elapsed_time]

        def set(self: TextGeneration.Step, d: Dict[str, Any]) -> None:
            """Append values to all non-wall_times fields in GeneratedRecord."""
            for k, v in d.items():
                if k in self._gr.stats:
                    self._gr.stats[k].append(v)
                else:
                    self._gr.stats[k] = [v]


@contextlib.contextmanager
def text_generation(ledger: Ledger) -> Generator[TextGeneration, None, None]:
    r = TextGeneration()
    yield r
    ledger.post(r.stats)


class Ledger:
    """Ledger consists of a list of JSON's persisting in a disk file."""

    def __init__(self: Ledger, *, prefix: str = "") -> None:
        self.folder = tempfile.TemporaryDirectory(prefix=prefix)
        self.fn = os.path.join(self.folder.name, "stats.jsonl")
        self.fp = open(self.fn, "w+")
        self.writer = jsonlines.Writer(self.fp)

    def __del__(self: Ledger) -> None:
        self.cleanup()

    def lock(self: Ledger) -> None:
        self.writer.close()  # No more posts.
        self.fp.close()

    def cleanup(self: Ledger) -> None:
        self.lock()
        self.fn = ""
        self.folder.cleanup()

    def post(self: Ledger, d: Dict[str, Any]) -> None:
        self.writer.write(d)  # Will except if locked.

    @contextlib.contextmanager
    def reader(self: Ledger) -> Generator[jsonlines.Reader, None, None]:
        self.lock()
        fp = open(self.fn, "r")
        rd = jsonlines.Reader(fp)
        yield rd
        rd.close()
        fp.close()


@dataclasses.dataclass
class PlotStats:
    sorted_steps: List[str]
    sorted_num_tokens_step: List[float]
    sorted_input_lengths: List[str]
    sorted_num_tokens_input_length: List[float]
    acceptance_rate: float


@dataclasses.dataclass
class TableStats:
    avg_token_per_step: float
    acceptance_rate: float
    avg_generate_candidates: float
    avg_verify_candidates: float
    avg_accept_candidate_tokens: float
    avg_step_time: float


def summarize(ledger: Ledger, batch_size: int, candidate_len: int) -> Tuple[TableStats, PlotStats]:
    THRESHOLD = 1

    steps: DefaultDict[str, int] = defaultdict(int)
    num_tokens_step: DefaultDict[str, float] = defaultdict(float)
    input_lengths: DefaultDict[str, int] = defaultdict(int)
    num_tokens_input_length: DefaultDict[str, float] = defaultdict(float)

    # Wall time stats
    generate_candidates: List[float] = []
    verify_candidates: List[float] = []
    accept_candidate_tokens: List[float] = []

    with ledger.reader() as reader:
        for line in reader:
            for step, num_tokens in zip(line["step"], line["avg_num_accepted"]):
                steps[step] += 1
                num_tokens_step[step] += batch_size * (float(num_tokens) + 1)
            for input_length, num_tokens in zip(line["input_id_length"], line["avg_num_accepted"]):
                input_lengths[input_length] += 1
                num_tokens_input_length[input_length] += batch_size * (float(num_tokens) + 1)
            generate_candidates += line["wall_times"]["generate_candidates"]
            verify_candidates += line["wall_times"]["verify_candidates"]
            accept_candidate_tokens += line["wall_times"]["accept_candidate_tokens"]

    sorted_steps = sorted(steps.keys())
    sorted_steps = [i for i in sorted_steps if steps[i] >= THRESHOLD]
    sorted_num_tokens_step = [num_tokens_step[i] / steps[i] for i in sorted_steps]

    sorted_input_lengths = sorted(input_lengths.keys())
    sorted_input_lengths = [i for i in sorted_input_lengths if input_lengths[i] >= THRESHOLD]
    sorted_num_tokens_input_length = [
        num_tokens_input_length[i] / input_lengths[i] for i in sorted_input_lengths
    ]

    total_steps = sum(steps[i] for i in sorted_steps)
    total_tokens = sum(num_tokens_step[i] for i in sorted_steps)
    avg_token_per_step = total_tokens / total_steps
    acceptance_rate = 100.0 * (avg_token_per_step - batch_size) / batch_size / candidate_len

    avg_generate_candidates = np.mean(np.array(generate_candidates))
    avg_verify_candidates = np.mean(np.array(verify_candidates))
    avg_accept_candidate_tokens = np.mean(np.array(accept_candidate_tokens))
    avg_step_time = avg_generate_candidates + avg_verify_candidates + avg_accept_candidate_tokens
    return TableStats(
        avg_token_per_step / batch_size,
        acceptance_rate,
        avg_generate_candidates,
        avg_verify_candidates,
        avg_accept_candidate_tokens,
        avg_step_time,
    ), PlotStats(
        sorted_steps,
        sorted_num_tokens_step,
        sorted_input_lengths,
        sorted_num_tokens_input_length,
        acceptance_rate,
    )


def draw_table(title: str, ts: TableStats) -> None:
    print(os.linesep)
    print(
        tabulate(
            [
                ["Average number of tokens per step:", f"{ts.avg_token_per_step:.4f}"],
                ["Average draft head acceptance rate (%):", f"{ts.acceptance_rate:.4f}"],
                SEPARATING_LINE,
                ["Wall time (seconds) per step", ""],
                ["", ""],
                ["generate_candidates:", f"{ts.avg_generate_candidates:.4f}"],
                [
                    "verify_candidates:",
                    f"{ts.avg_verify_candidates:.4f}",
                ],
                ["accept_candidate_tokens:", f"{ts.avg_accept_candidate_tokens:.4f}"],
                SEPARATING_LINE,
                ["Tokens/second", f"{ts.avg_token_per_step / ts.avg_step_time:.4f}"],
                SEPARATING_LINE,
            ],
            headers=[f"Title: {title}", ""],
            colalign=("left", "right"),
        )
    )


def draw_figure(title: str, ps: PlotStats, filename: str) -> None:
    pass
