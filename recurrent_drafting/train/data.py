#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
from typing import Any, Dict, List

import fastchat.model
import torch
import transformers

# This special token is supposed to appear in the target of training instances. The loss function
# doesn't take it into consideration.
IGNORE_TOKEN_ID = transformers.trainer_pt_utils.LabelSmoother.ignore_index


def _sharegpt_conversation_to_vicuna_prompt(
    conversation: List[Dict[str, Any]], tmplt: fastchat.conversation.Conversation
) -> str:
    roles = {"human": tmplt.roles[0], "gpt": tmplt.roles[1]}  # human->USER, gpt->ASSISTANT
    tmplt.messages = []
    for i, turn in enumerate(conversation):
        assert turn["from"] == ["human", "gpt"][i % 2], "human and gpt must take turns"
        tmplt.append_message(roles[turn["from"]], turn["value"])
    return tmplt.get_prompt()


def _create_labels(
    prompt: str,
    input_ids: torch.Tensor,
    tmplt: fastchat.conversation.Conversation,
    tokenizer: transformers.PreTrainedTokenizer,
) -> torch.Tensor:
    """Construct labels given the prompt string and related token IDs. For the following example
    prompt, it returns a labels tensor, where ignorable tokens are masked -100.

    "{system_prompt} USER: {u1} ASSISTANT: {a1}</s>USER: {u2} ASSISTANT: {a2}</s> ...</s>"
    [ -100           ...             -100  {a1}</s> -100    ...    -100  {a2}</s> ...</s>]

    """
    labels = input_ids.clone()
    sep = tmplt.sep + tmplt.roles[1] + ": "  # " ASSISTANT: "
    n_tokens = int(labels.ne(tokenizer.pad_token_id).sum())
    turns = prompt.split(tmplt.sep2)  # sep2 = "</s>"
    cur_len = 1
    labels[:cur_len] = IGNORE_TOKEN_ID
    for i, turn in enumerate(turns):
        if turn == "":
            break
        turn_len = len(tokenizer(turn).input_ids)
        parts = turn.split(sep)
        if len(parts) != 2:
            break
        parts[0] += sep
        # "-2" is hardcoded for the LLaMA tokenizer to make the offset correct.
        instruction_len = len(tokenizer(parts[0]).input_ids) - 2
        # Ignore the user instructions
        labels[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
        cur_len += turn_len
    labels[cur_len:] = IGNORE_TOKEN_ID
    if cur_len < tokenizer.model_max_length:
        if cur_len != n_tokens:
            labels[:] = IGNORE_TOKEN_ID
    return labels


def sharegpt_record_to_vicuna_training_instance(
    sharegpt_record: Dict[str, Any],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict[str, torch.Tensor]:
    """Given a conversation, outputs a training instance into which the conversation is
    converted.

    Returns:

      A dictionary of tokenized inputs, labels, and attention mask.

    """
    # We have to use the conversation template vicuna for the fine-tuning of the drafter, because
    # the base model LLaMA was trained using this template. The following is an example
    # prompt. Please be aware of the leading <s> and the </s> at the end of each turn.
    #
    # "<s> A chat between a curious user and an artificial intelligence assistant. "
    # "The assistant gives helpful, detailed, and polite answers to the user's questions. "
    # "USER: How are you? ASSISTANT: I'm good. How are you?</s>"
    # "USER: I am good too.</s>"
    conversation = sharegpt_record["conversations"]

    tmplt = fastchat.model.model_adapter.get_conversation_template("vicuna")
    # tmplt.roles[0] == 'USER
    # tmplt.roles[1] == 'ASSISTANT'
    # tmplt.sep == ' ', the space before "USER" and "ASSISTANT"
    # tmplt.sep2 == '</s>', the separator between turns

    if conversation[0]["from"] != "human":  # Skip the first one if it is not from human
        conversation = conversation[1:]

    prompt = _sharegpt_conversation_to_vicuna_prompt(conversation, tmplt)

    input_ids = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids[0]

    labels = _create_labels(prompt, input_ids, tmplt, tokenizer)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": input_ids.ne(tokenizer.pad_token_id),
    }


def convert_alpaca_to_sharegpt(d: Dict[str, str]) -> Dict[str, Any]:
    return {
        "id": "0",
        "conversations": [
            {"from": "human", "value": d["instruction"]},
            {"from": "gpt", "value": d["output"]},
        ],
    }
