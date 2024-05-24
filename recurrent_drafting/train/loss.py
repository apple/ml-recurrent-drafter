#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
from torch.nn import CrossEntropyLoss

from recurrent_drafting.train.data import IGNORE_TOKEN_ID


def drafter_loss(logits, labels, next_n, top_k):
    # logits: [drafter_predict_n_tokens, batch_size, seq_len, vocab_size]
    # labels: [batch_size, seq_len]

    # Stacked logits in the drafter's forward function.
    # [[[l2]_1, [l3]_1, ..., [l_{n+1}]_1], <- first logit predicts token 2
    #  [[l3]_2, [l4]_2, ..., [l_{n+2}]_2], <- first logit predicts token 3
    #  ...
    loss, log, eval_log = 0, {}, {}
    for i in range(next_n):
        logits_i = logits[i, :, : -(i + 2)].contiguous().view(-1, logits.shape[-1])
        labels_i = labels[..., i + 2 :].contiguous().view(-1).to(logits_i.device)
        loss_i = CrossEntropyLoss()(logits_i, labels_i)
        loss += loss_i
        not_ignore = labels_i.ne(IGNORE_TOKEN_ID)
        labels_i = labels_i[not_ignore]
        # Add top-k accuracy
        for k in range(1, top_k + 1):
            topk = logits_i.topk(k, dim=-1)[-1][not_ignore]
            correct = topk.eq(labels_i.unsqueeze(-1)).any(-1)
            log[f"redrafter{i}_top{k}"] = correct.float().mean().item()
            eval_log[f"redrafter{i}_top{k}"] = correct.float().mean()
        log[f"redrafter{i}_loss"] = loss_i.item()
    return loss, log, eval_log
