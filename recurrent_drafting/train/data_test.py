#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import torch
import transformers

from . import data

_TEST_SHAREGPT_RECORD = {
    "id": "0",
    "conversations": [
        {
            "from": "human",
            "value": "What are some famous actors that started their careers on Broadway?",
        },
        {
            "from": "gpt",
            "value": "Some famous actors that started their careers on Broadway"
            " include Hugh Jackman",
        },
    ],
}


def test_sharegpt_conversations_to_vicuna_training_instance() -> None:
    prompt_prefix = (
        "<s> A chat between a curious user and an artificial intelligence assistant. "
        + "The assistant gives helpful, detailed, and polite answers to the user's questions. "
    )
    expected_prompts = (
        prompt_prefix
        + "USER: What are some famous actors that started their careers on Broadway? ASSISTANT: "
        + "Some famous actors that started their careers on Broadway include Hugh"
    )
    expected_converted_labels = (
        "<unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk>"
        + "<unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk>"
        + "<unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk>"
        + "<unk> Some famous actors that started their careers on Broadway include Hugh"
    )

    seq_length = 64
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "lmsys/vicuna-7b-v1.3",
        model_max_length=seq_length,
        padding_side="right",
    )

    ret = data.sharegpt_record_to_vicuna_training_instance(_TEST_SHAREGPT_RECORD, tokenizer)
    assert ret["input_ids"].shape == (seq_length,)
    assert ret["labels"].shape == (seq_length,)
    assert ret["attention_mask"].shape == (seq_length,)
    assert ret["input_ids"].ne(0).sum(dim=-1).tolist() == ret["attention_mask"].sum(dim=-1).tolist()

    decoded_inputs = tokenizer.decode(ret["input_ids"])
    assert decoded_inputs == expected_prompts

    decoded_labels = tokenizer.decode(torch.relu(ret["labels"]))
    assert decoded_labels == expected_converted_labels

    assert ret["attention_mask"].sum(dim=-1) == seq_length - decoded_inputs.count("<unk>")


_TEST_ALPACA_RECORD_DICT = {
    "dataset": "helpful_base",
    "instruction": "What are some famous actors that started their careers on " "Broadway?",
    "output": "Some famous actors that started their careers on Broadway include" "Hugh Jackman",
    "generator": "text_davinci_003",
}


def test_alpaca_to_sharegpt_convert():
    sharegpt_record = data.convert_alpaca_to_sharegpt(_TEST_ALPACA_RECORD_DICT)
    assert sharegpt_record["conversations"][0]["from"] == "human"
    assert sharegpt_record["conversations"][0]["value"] == _TEST_ALPACA_RECORD_DICT["instruction"]
    assert sharegpt_record["conversations"][1]["from"] == "gpt"
    assert sharegpt_record["conversations"][1]["value"] == _TEST_ALPACA_RECORD_DICT["output"]
