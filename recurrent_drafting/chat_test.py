from typing import Any, Callable, Dict, List

import datasets
import transformers

from . import chat

_TEST_SHAREGPT_RECORD: Dict[str, Any] = {
    "id": "0",
    "conversations": [
        {"from": "human", "value": "Hello"},
        {"from": "gpt", "value": "World"},
        {"from": "human", "value": "Oops"},
        {"from": "gpt", "value": "Bingo Bingo"},
    ],
}


_TEST_SWIFT_ASSIST_RECORD: Dict[str, Any] = {
    "messages": [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "World"},
        {"role": "user", "content": "Oops"},
        {"role": "assistant", "content": "Bingo Bingo"},
    ]
}

_TEST_STANDARD_RECORD = ["Hello", "World", "Oops", "Bingo Bingo"]


def test_standardize_sharegpt():
    chat.standardize_sharegpt(_TEST_SHAREGPT_RECORD) == _TEST_STANDARD_RECORD


def test_vicuna_prompt():
    x = chat.vicuna_prompt(_TEST_STANDARD_RECORD)
    print(x)
    assert (
        x
        == "A chat between a curious user and an artificial intelligence assistant. The assistant"
        + " gives helpful, detailed, and polite answers to the user's questions. USER: Hello"
        + " ASSISTANT: World</s>USER: Oops ASSISTANT: Bingo Bingo</s>"
    )


def test_find_all_sublists():
    long_list = ["a", "b", "c", "d", "e", "b", "c", "h"]
    short_list = ["b", "c"]
    idx = chat.find_all_sublists(long_list, short_list)
    assert idx == [1, 5]


def _unignorable_tokens(labels: List[int]) -> List[int]:
    return [e for e in labels if e != chat.IGNORE_TOKEN_ID]


def test_vicuna_labels():
    prompt = chat.vicuna_prompt(_TEST_STANDARD_RECORD)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "recurrent_drafting/testdata/vicuna-7b-v1.3-tokenizer",
        padding_side="right",
    )
    # Test truncating before each "<turn_end>"
    token_list = tokenizer.convert_ids_to_tokens(tokenizer(prompt).input_ids)
    user_ends = chat.find_all_sublists(token_list, [r"▁A", r"SS", r"IST", r"ANT", r":"])
    model_ends = chat.find_all_sublists(token_list, ["</s>"])
    ends = sorted(user_ends + model_ends)
    ends.append(ends[-1] + 5)  # +1 includes the last <turn_end>
    for ith_turn, max_length in enumerate(ends):
        ids = tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True,
        ).input_ids[0]
        labels = chat.vicuna_labels(ids, tokenizer)
        label_ids = _unignorable_tokens(labels.tolist())
        assert (
            tokenizer.convert_ids_to_tokens(label_ids) == []
            if ith_turn <= 1
            else (
                [r"▁World", r"</s>"]
                if ith_turn == 2
                else [r"▁World", r"</s>", r"▁B", r"ingo", r"▁B", r"ingo", r"</s>"]
            )
        )


def test_sharegpt_to_vicuna():
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "recurrent_drafting/testdata/vicuna-7b-v1.3-tokenizer",
        padding_side="right",
        model_max_length=2048,  # The typical length we used to real training jobs.
    )

    def process_dataset_for_the_first_labels(
        preprocessor: Callable[[Dict[str, Any], transformers.PreTrainedTokenizer], Dict[str, Any]]
    ) -> Dict[str, Any]:
        data_list = [
            e
            for e in datasets.load_dataset(
                "recurrent_drafting/testdata/sharegpt_tiny", split="train"
            ).map(lambda x: preprocessor(x, tokenizer))
        ]
        assert {"input_ids", "labels", "attention_mask"}.issubset(data_list[0].keys())
        return data_list[0]

    y = process_dataset_for_the_first_labels(chat.sharegpt_for_vicuna)
    assert all(
        a == b or b == -100 for a, b in zip(y["input_ids"], y["labels"])
    ), "Elements are not equal or one is not -100"


def test_sharegpt_dataset_for_vicuna_generation():
    def vicuna_user_prompt(vicuna_record: Dict[str, Any]) -> Dict[str, Any]:
        conversation = chat.standardize_sharegpt(vicuna_record)
        return {"prompt": chat.vicuna_prompt(conversation, add_generation_prompt=True)}

    prompt_list = [
        e["prompt"]
        for e in datasets.load_dataset(
            "recurrent_drafting/testdata/sharegpt_tiny", split="train"
        ).map(vicuna_user_prompt)
    ]
    assert prompt_list == [
        chat.VICUNA_SYSTEM_PROMPT
        + "USER: Summarize the main ideas of Jeff Walker's Product Launch Formula into bullet "
        + "points as it pertains to a growth marketing agency implementing these strategies and"
        + " tactics for their clients... ASSISTANT:"
    ]
