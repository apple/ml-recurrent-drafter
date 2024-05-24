## Convert a Beam Into a Prefix Tree

Consider that for a context, the draft model returns a beam of three candidate token sequences.

``` python
import torch

beam = torch.tensor(
    [
        [101, 102, 103, 104],  # Mars is a       red
        [101, 102, 203, 204],  # Mars is reddish when
        [101, 102, 303, 104],  # Mars is dark    red
    ]
)
```

The following line uses the trick of [Pairwise Comparison of Low-dimensional Embeddings In A Tensor](pairwise_comparison.md) to compare sequences pairwisely.

```python
matches = beam[None, :, :] == beam[:, None, :]

# tensor([[[ True,  True,  True,  True],
#          [ True,  True, False, False],
#          [ True,  True, False,  True]],
#
#         [[ True,  True, False, False],
#          [ True,  True,  True,  True],
#          [ True,  True, False, False]],
#
#         [[ True,  True, False,  True],
#          [ True,  True, False, False],
#          [ True,  True,  True,  True]]])
```

The first matrix in `matches` comparisons the first sequence in the beam with all sequences in the beam.  The first row contains four True's because the first sequence is idential to itself.  The second row ends with two False's because "reddish" and "when" differ from "a" and "red".

Similarly, the second and the third matrices in `matches` are pairwise comparison of the second and the third sequences in the beam with all sequences.

In the above example, `matches` is the result of token-wise comparisons; however, as discussed in [Tree Attention](tree_attention.md), we want prefix-wise comparison.  For example, the bottom-right True in the first matrix of `matches` indicates that both the first and the third sequences end with "red".  We want that value to be False instead, because the two "red"s have different prefixes: the first has "Mars is a" and the second has "Mars is dark".

In order to convert the token-wise comparison result into prefix-wise, we utilizes the property that the left-most False in each row indicates that all tokens to its right do not share the prefix with the reference sequence.  The following code uses the trick of cumsum to reset all True's to the right of a False.

```python
prefix_target = torch.arange(beam.shape[-1]) + 1

# tensor([1, 2, 3, 4])

prefix_matches = torch.cumsum(matches, dim=-1) == prefix_target[None, None, None, :]

# tensor([[[[ True,  True,  True,  True],
#           [ True,  True, False, False],
#           [ True,  True, False, False]],
#
#          [[ True,  True, False, False],
#           [ True,  True,  True,  True],
#           [ True,  True, False, False]],
#
#          [[ True,  True, False, False],
#           [ True,  True, False, False],
#           [ True,  True,  True,  True]]]])
```

Next, for each of the three matrices in `prefix_matches`, locate the first True in each column. The locations represent the prefix tree, as specified in [Tre Attention](tree_attention.md).

To demonstrate how this works, consider the second matrix in `prefix_matches`.  The second matrix compares sequence 1 to all three sequences. Row 1 of this matrix contains all True values as it compares sequence 1 to itself, as expected.

All three sequences share the first two tokens, hence the first two columns of this matrix contain all True values. For these two columns, the first True appears at location 0, because the first two nodes of branch 1 are from sequence 0, "Mars is".  The first True appearance in the remaining two columns occurs at location 1, as the last two nodes of branch 1 are from sequence 1, "reddish when".

We can use `argmax(dim=-2)` to locate the first True in each column.  To use argmax, we must convert `prefix_matches` to integer because `argmax` does not accept boolean tensors.

```python
prefix_matches.to(torch.int32)

# tensor([[[[1, 1, 1, 1],
#           [1, 1, 0, 0],
#           [1, 1, 0, 0]],
#
#          [[1, 1, 0, 0],
#           [1, 1, 1, 1],
#           [1, 1, 0, 0]],
#
#          [[1, 1, 0, 0],
#           [1, 1, 0, 0],
#           [1, 1, 1, 1]]]], dtype=torch.int32)

print(torch.argmax(prefix_matches.to(torch.int32), dim=-2))

# tensor([[[0, 0, 0, 0],
#          [0, 0, 1, 1],
#          [0, 0, 2, 2]]])
```

In practice, the input tensor `beam` is three-dimensional because the first dimension is the batch.  Assume the batch size is 2 and both beams are identical.

```python
beam = torch.tensor(
    [
        [
            [101, 102, 103, 104],  # Mars is a       red
            [101, 102, 203, 204],  # Mars is reddish when
            [101, 102, 303, 104],  # Mars is dark    red
        ],
        [
            [101, 102, 103, 104],  # Mars is a       red
            [101, 102, 203, 204],  # Mars is reddish when
            [101, 102, 303, 104],  # Mars is dark    red
        ],
    ]
)
```

The only change required is to incorporate the batch dimension when computing matches.

```python
matches = beam[:, None, :, :] == beam[:, :, None, :]
print(matches)
```
