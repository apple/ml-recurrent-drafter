#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import time
from typing import Any, Dict, List

import recurrent_drafting


def test_ledger() -> None:
    ledger = recurrent_drafting.stats.Ledger()
    data: List[Dict[str, Any]] = [{"apple": 1, "orange": 2}, {}, {"banana": 3}]
    for d in data:
        ledger.post(d)
    with ledger.reader() as reader:
        i = 0
        for j in reader:
            assert j == data[i]
            i += 1


def test_text_generate_step() -> None:
    g = recurrent_drafting.stats.TextGeneration()
    for _ in range(3):
        with g.step() as s:
            assert isinstance(s, recurrent_drafting.stats.TextGeneration.Step)
            with s.time("verify_candidates"):
                time.sleep(0.001)
            s.set({"input_id_length": 10, "avg_num_accepted": 5})
    assert g.n_steps == 3
    assert g.stats["step"] == [0, 1, 2]
    assert g.stats["input_id_length"] == [10, 10, 10]
    assert g.stats["avg_num_accepted"] == [5, 5, 5]
    assert len(g.stats["wall_times"]["verify_candidates"]) == 3
    assert g.stats["wall_times"]["verify_candidates"][0] > 0
    assert g.stats["wall_times"]["verify_candidates"][1] > 0
    assert g.stats["wall_times"]["verify_candidates"][2] > 0


def test_text_generation() -> None:
    ledger = recurrent_drafting.stats.Ledger()
    with recurrent_drafting.stats.text_generation(ledger) as r:
        assert isinstance(r, recurrent_drafting.stats.TextGeneration)
        assert r.n_steps == 0
        assert len(r.stats["step"]) == 0
    with ledger.reader() as reader:
        rs = [r for r in reader]
        assert len(rs) == 1
        assert rs[0] == {"step": [], "wall_times": {}}


def test_summarize() -> None:
    def fake_generate(gr: recurrent_drafting.stats.TextGeneration) -> None:
        for step in range(3):
            with gr.step() as sr:
                with sr.time("generate_candidates"):
                    time.sleep(0.001)
                with sr.time("verify_candidates"):
                    time.sleep(0.001)
                with sr.time("accept_candidate_tokens"):
                    time.sleep(0.001)
                sr.set({"input_id_length": 10 + 6 * step, "avg_num_accepted": 5})

    ledger = recurrent_drafting.stats.Ledger()
    for generation in range(2):
        with recurrent_drafting.stats.text_generation(ledger) as gr:
            fake_generate(gr)

    ts, _ = recurrent_drafting.stats.summarize(ledger, 2, 5)
    assert ts.acceptance_rate == 100.0
    recurrent_drafting.stats.draw_table("candidate_length=5,batch_size=2", ts)
