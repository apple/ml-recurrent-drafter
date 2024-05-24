# Tree Attention

## Motivation and Concepts

We can use the draft model to run the beam search algorithm and get a beam of M candidate token sequences, each with C tokens. As an example,

```
Mars is a       red
Mars is reddish when
Mars is dark    red
```

We can then use the LLM to rate this beam of candidate tokens and accept the first few in sequence. However, duplicated tokens across sequences, such as "Mars" and "is", would waste some flops.

Unfortunately, duplication is unavoidable because the draft model is likely to be more predictable with the first few tokens than the others. As a result, the first tokens in these sequences are very often duplicated. Mars is the first token in each of the three sequences in the beam above. The final tokens in these sequences are distinctive, red, from, and dark.

To reduce the computational cost, we aim to reduce duplications. A straightforward idea is to remove duplicate tokens. However, it breaks the attention causality. For example, the LLM's self-attention layers make use of the fact that the token "red" in the first sequence attends to "a" while the "red" in the last sequence attends to "dark". If we deduplicate the two "red"s, this united token can only attend to "a" or "dark", but not both. This is not what we wanted.

To retain attention causality, we should deduplicate prefixes rather than tokens. For example, we may deduplicate all three "is"s because they have the same prefix "Mars".  In other words, when the LLM computes the hidden state of "is", it may reuse the keys and values of "Mars".

The prefix tree is an intuitive way for illustrating common prefixes. The above beam of candidate tokens could be restructured into the following tree, with each branch representing a sequence.

```
Mars - is - a ------- red
         \- reddish - when
         \- dark    - red
```

After deduplicating common prefixes, we get the following tokens, which correspond to the tokens in the prefix tree shown above.

```
Mars is a red reddish when dark red
```

The attention causality is as follows. It is the exact same prefix tree as shown above!

```
Mars ← is ← a ← red reddish ← when dark ← red
        ↖----------/              /
          ↖---------------------/
```

The GPU only accepts tensors, thus we must describe the above deduplication result and attantion causality as tensors. Tokens is a vector containing token IDs. The causality is a two-dimensional matrix like the following:

```
         Mars  is   a  red reddish when dark red
Mars      Y
is        Y    Y
a         Y    Y    Y
red       Y    Y    Y    Y
reddish   Y    Y              Y
when      Y    Y              Y    Y
dark      Y    Y                        Y
red       Y    Y                        Y    Y
```

## From Beam to Prefix Tree

We represent a prefix tree by tensors and develop a GPU-friendly algorithm to convert a beam into a prefix tree.

With our representation, the above prefix tree is encoded by the following tensor, whose each row encodes a branch of the prefix tree, and each branch corresponds to a sequence in the beam.

```
0 0 0 0
0 0 1 1
0 0 2 2
```

The four 0's in row 0 indicate that all four nodes of branch 0 originate from sequence 0 in the beam, namely "Mars is a red". The two 1s in row 1 indicate that the final two nodes of branch 1 are from sequence 1, specifically "reddish when". The two 0s indicate that the first two nodes of branch 1 are from sequence 0, as sequences 0 and 1 share the prefix "Mars is".  Similarly, the placements of 2's in row 2 show tokens in sequence 2 that do not share prefixes with their counterparts in other sequences.

The function `tree_attention._dedup_prefix` converts a beam into the above encoding of the prefix tree. The notebook docs/beam_to_prefix_tree.py explains this function.

## Pack the Prefix Tree

The aforementioned prefix tree encoding allows us to select tokens from `beam` for verification. For the $i$-th sequence, we search for the value $i$ in the above encoding. The token in `beam` that appears where the value $i$ appears in the prefix tree encoding is the one that needs to be verified and hence packed.

For example, in row 0 of the prefix tree encoding, all four elements are zero. This means we must select all four tokens from the beam's first sequence: "Mars", "is", "a", and "red". Only the last two elements in row 1 of the prefix tree encoding are 1, so we select the final two tokens from the second sequence, "reddish" and "when". Similarly, we selected "dark" and "red" from the third sequence.

The function `_pack_beams` converts each beam in a batch sequence to a one-dimensional list of `n_selected` token IDs:

```
["Mars", "is", "a", "red", "reddish", "when", "dark", "red"]
```

It also provides an index vector, `packed_token_indices`, for each sequence, which indicates where the tokens were chosen inside the beam. Because "Mars" is the first token in the initial sequence, its index is zero. "Reddish" is the third token in the second sequence; its index is 4+2=6, with 4 representing the beam length. The chosen tokens' indexes are

```
[0, 1, 2, 3, 6, 7, 10, 11]
```

## The Unpacker

The unpacker is an index tensor that turns the output of `_pack_beams` back to the beam.  For example, to convert the above-mentioned output

```
["Mars", "is", "a", "red", "reddish", "when", "dark", "red"]
```

back to the beam of candidate tokens

```
Mars is a       red
Mars is reddish when
Mars is dark    red
```

we need the unpacker

```
0 1 2 3
0 1 4 5
0 1 6 7
```

The unpacker's shape is identical to the beam. The unpacker's elements index the packed beam.

The function `_get_unpacker` creates the unpacker using the beam and its prefix tree encoding. The algorithm is based on the fact that each candidate token sequence in a beam comprises two segments: the prefix from the preceding sequence and the rest.

The prefix tree encoding reveals the border between each pair of segments.  In the preceding case, we have:

```
| 0 0 0 0
0 0 | 1 1
0 0 | 2 2
```

Marking the aforementioned border in the beam, we obtained:

```
| Mars is a       red
Mars is | reddish when
Mars is | dark    red
```

Tokens from each segment appear sequentially in the packed beam.  For example, the first sequence in a beam contains no prefix segment from prior sequences, therefore all tokens in the first sequence are to the right of the segment border. "Mars", "is", "a", and "red" are the first four tokens in the packed beam. Similarly, the second segment of the third candidate sequences, "dark" and "red", occur sequentially near the end of the packed beam.

This attribute implies that we design the unpacker in two steps.  The first step locates the start of each segment in the packed beam. For example:

```plaintext
[0] 0  0  0
[0] 0 [4] 4
[0] 0 [6] 6
```

Please keep in mind that the first segment of each candidate token sequence begins at position 0 in the packed beam.

The second step adds consecutive offsets within the corresponding segment

```plaintext
0 1 2 3                0 1 2 3
0 1 0 1    and gets    0 1 4 5
0 1 0 1                0 1 6 7
```

For further information on the preceding two steps, please see the comments in the source code of `_get_unpacker`.

## API Design

We expect that the text generation algorithm look like the following:

```python
beam, scores = draft_candidate_tokens(drafter)
packed_beam, attention_mask, position_offsets, unpack_map = tree_attention.pack(beam)
out = LLM(packed_beam, attention_mask, position_offsets)
verified_beam, verified_scores = tree_attention.unpack(out, unpack_map)
tokens = accept_candidate_tokens(beam, scores, verified_beam, verified_scores)
```

- `beam` contains token IDs and is in shape $(B, M, C)$, where B is batch size, $M$ is the number of candidate token sequences, and $C$ is the sequence legnth.
- `scores` has the same size as `beam` but includes scores of the candidate tokens.
- `packed_beam` is the result from prefix-deduplication and is in shape $(B, L)$, where $L\\leq M\\times C$.
- `attention_mask` is a boolean tensor in shape $(B, L, L)$.
- `position_offsets` is an integer tensor in shape $(B, L)$ where each element tells how far away each candidate token is from the end of the processed context.
- `unpack_map` is an integer tensor in shape $(B, M, C)$ where each element is in range $\[0, L)$. It has the property `packed_beam[unpack_map] == beam`.
- `verified_beam` has the same shape as `beam`.
- `verfied_scores` has the same shape as `scores`.

The following figure illustrates the expected usage of the API design:

<center><img width=50% src=tree_attention.png /></center>
