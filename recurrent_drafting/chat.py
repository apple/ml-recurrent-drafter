"""This file defines the code that works with HuggingFace datasets to convert training and
evaluation data from JSON into a dictionary with the following components:

1. input_ids: torch.Tensor(int32) - Right-padded input token IDs.
2. labels: torch.Tensor(int32) - Right-padded labels with instructions masked by IGNORE_TOKEN_ID.
3. padding_mask: torch.Tensor(bool) - Indicates where input_ids are equal to PAD_TOKEN.

The design aims to support various JSON schemas, tokenizers, special tokens of different models, and
chat templates of the models. To maximize code reuse, we decompose the conversation processing into
the following phases:


### Phase 1: Standardize JSON Schemas

Various models have their own chat templates and special tokens. To decouple datasets from specific
models, dataset providers tend to save data in JSON format as a list of conversation turns rather
than a string representation of the conversation. This is because converting a conversation into a
string requires the chat template of a specific model.

Different datasets have their own JSON schemas. For example, each ShareGPT conversation is formatted
as follows:

{
    "id": "0",
    "conversations": [
        {"from": "human", "value": "Hello"},
        {"from": "gpt", "value": "World"},
        {"from": "human", "value": "Oops"},
        {"from": "gpt", "value": "Bingo"}
    ]
}

In contrast, each conversation is formatted like this:

{
    "messages": [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "World"},
        {"role": "user", "content": "Oops"},
        {"role": "assistant", "content": "Bingo"}
    ]
}

Although the content is the same, the JSON schemas differ. Therefore, it makes sense to convert them
into a standardized Python format:

turns = [
    "Hello",
    "World",
    "Oops",
    "Bingo"
]


### Phase 2: Convert Standardized Schema to Text

To train a Vicuna model, we need to convert the conversation into the following string:

"A chat between a curious user and an artificial intelligence assistant. " +
"The assistant gives helpful, detailed, and polite answers to the user's questions. " +
"USER: Hello " +
"ASSISTANT: World</s>" +
"USER: Oops " +
"ASSISTANT: Bingo</s>"

The first two lines in the above string literal are the system prompt for the Vicuna model.


### Phase 3: Convert Tokenized Text into Label IDs

By tokenizing the above text, we obtain input token IDs for training the model. However, we also
need the label token IDs.

The goal of training a draft model is to ensure that given a user prompt, the model generates
answers similar to those produced by the LLM. To achieve this, we craft the label token ID vector by
masking tokens in the system prompt, encapsulators, and user prompts with IGNORE_TOKEN_ID.

In the following example, we mark tokens to be masked out with 'x' and those to be kept with 'o':

"A chat between a curious user and an artificial intelligence assistant. " +
 x x    x       x x       x    x   x  x          x            x        xx
"The assistant gives helpful, detailed, and polite answers to the user's questions. " +
 x   x         x     x      x x       x x   x      x       x  x   x      x        xx
"USER: Hello " +
 x   xxx    x
"ASSISTANT: World</s>" +
 x        xx o   o
"USER: Oops " +
 x   xxx   x
"ASSISTANT: Bingo</s>" +
 x        xx o   o

This approach ensures that the model focuses on generating relevant responses while ignoring
system-specific and user prompt tokens."""

from typing import Any, Dict, List

import torch
import transformers

# This special token is supposed to appear in the target of training instances. The loss function
# doesn't take it into consideration.
IGNORE_TOKEN_ID = transformers.trainer_pt_utils.LabelSmoother.ignore_index


def standardize_sharegpt(record: Dict[str, Any]) -> List[str]:
    assert all(
        turn["from"] == "human" if index % 2 == 0 else "gpt"
        for index, turn in enumerate(record["conversations"])
    )
    return [turn["value"] for turn in record["conversations"]]


VICUNA_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    + "The assistant gives helpful, detailed, and polite answers to the user's questions. "
)


def vicuna_prompt(turns: List[str], add_generation_prompt: bool = False) -> str:
    def encap(index: int, turn: str) -> str:
        left = "USER: " if index % 2 == 0 else "ASSISTANT: "
        right = " " if index % 2 == 0 else "</s>"
        return left + turn + right

    return (
        VICUNA_SYSTEM_PROMPT + encap(0, turns[0]) + "ASSISTANT:"
        if add_generation_prompt
        else VICUNA_SYSTEM_PROMPT + "".join(encap(index, turn) for index, turn in enumerate(turns))
    )


def find_all_sublists(lst, sublst):
    len_lst = len(lst)
    len_sublst = len(sublst)
    indices = []
    for i in range(len_lst - len_sublst + 1):  # This is low-efficient O(N M), but concise.
        if lst[i : i + len_sublst] == sublst:
            indices.append(i)
    return indices


def vicuna_labels(ids: torch.Tensor, tokenizer: transformers.PreTrainedTokenizer) -> torch.Tensor:
    assert ids.dim() == 1  # A batch of one tokenized text
    token_list = tokenizer.convert_ids_to_tokens(ids.tolist())
    user_ends = find_all_sublists(token_list, ["▁A", "SS", "IST", "ANT", ":"])
    user_ends = [e + 5 for e in user_ends]  # shift to the end of " ASSISTANT:"
    model_ends = find_all_sublists(token_list, ["</s>"])
    assert len(user_ends) - len(model_ends) in [0, 1]  # truncation may loss the last </s>
    # len(model_ends) <= 0 indicates truncated the first model response. Mask out all input_ids.
    if len(model_ends) <= 0:
        return torch.full(ids.size(), IGNORE_TOKEN_ID)
    labels = ids.clone()
    start = 0
    for i in range(len(model_ends)):
        labels[start : user_ends[i]] = IGNORE_TOKEN_ID
        if i < len(model_ends):  # guards len(model_ends) = len(user_ends)-1
            start = model_ends[i] + 1  # skip and keep </s>
    labels[model_ends[-1] + 1 :] = IGNORE_TOKEN_ID
    return labels


def sharegpt_for_vicuna(
    sharegpt_record: Dict[str, Any], tokenizer: transformers.PreTrainedTokenizer
) -> Dict[str, Any]:
    ids = tokenizer(
        vicuna_prompt(standardize_sharegpt(sharegpt_record)),
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids[0]
    return {
        "input_ids": ids,
        "labels": vicuna_labels(ids, tokenizer),
        "attention_mask": ids.ne(tokenizer.pad_token_id),
    }
