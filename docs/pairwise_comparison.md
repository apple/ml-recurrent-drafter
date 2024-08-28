# Pairwise Comparison of Low-dimensional Embeddings In A Tensor

Building the tree attention requires organizing a beam of candidate token sequences into a prefix
tree so to remove redundant common prefixes.  This invention relies on a widely applicable trick:
[pairwise comparison of elements, rows, etc. in a tensor](https://shorturl.at/cHU27). This notebook
explains this trick.

## Pairwise Compare Elements in a Vector

The most basic form of comparison is when the input is a vector of elements. Assume we want to
compare pairwise elements in the following vector l.

```python
import torch

b = torch.tensor([1, 2, 3, 3])
```

We could imagine that the result should be the following:

```
    1     2      3      3
 [[True, False, False, False],  1
  [False, True, False, False],  2
  [False, False, True, True],   3
  [False, False, True, True]]   3
```

To achieve the desired result, we must compare the first element 1 with all of the elements
\[1,2,3,3\], the second element 2 with \[1,2,3,3\], the third element 3 with \[1,2,3,3\], and the
fourth element 3 (again) with \[1,2,3,3\].

If we construct the above four comparisons in matrix form, we can compare

```
X = [[1, 2, 3, 3],
     [1, 2, 3, 3],
     [1, 2, 3, 3],
     [1, 2, 3, 3]]
```

with

```
Y = [[1, 1, 1, 1],
     [2, 2, 2, 2],
     [3, 3, 3, 3],
     [3, 3, 3, 3]]
```

The matrix x results from broadcasting the input tensor.

```
X = [[1, 2, 3, 3],
      .  .  .  .
      .  .  .  .
      .  .  .  .]]
```

The matrix y is the transpose of the input vector broadcasted right.

```
Y = [[1], ...
     [2], ...
     [3], ...
     [3], ... ]
```

To prepare for broadcasting of the input tensor, we must add a new dimension as the first
dimension. So, broadcasting in this dimension replicates the vector. To add this new first
dimension, we can use the PyTorch syntax b\[None, :\].

```python
x = b[None, :]  # tensor([[1, 2, 3, 3]])
```

Please be aware that b\[:\] is exactly l.

To prepare for broadcasting elements of the input vector, we must add a new dimension as the new
last dimension.

```python
y = b[:, None]

# tensor([[1],
#         [2],
#         [3],
#         [3]])
```

When you compare x and y in PyTorch, they will be broadcast automatically. To allow comparison of
the first (and only) row of x, \[1,2,3,3\], with the first element of y, \[1\], the automatic broadcast
extends \[1\] into \[1,1,1,1\], which is the first row of Y.

To allow comparison with the all the element of y,

```
[[1]
 [2]
 [3]
 [4]]
```

the automatic broadcast extends the first (and only) row of x, \[1,2,3,3\], into

```
[[1,2,3,3],
 [1,2,3,3],
 [1,2,3,3],
 [1,2,3,3]]
```

which is X.

```python
print(x == y)

# tensor([[ True, False, False, False],
#         [False,  True, False, False],
#         [False, False,  True,  True],
#         [False, False,  True,  True]])
```

## Pairwise Compare Rows in a Matrix

Let us expand the cast from one-dimensional into two-dimensional.  Consider that we want pariwise
comparison of the three rows of b.

```python
b = torch.tensor([[101, 102, 103, 104], [101, 102, 203, 204], [101, 102, 303, 104]])
```

Let us imagine what we will get from this pairwise comparison:

1. When we compare the first row to itself or any other row, we should get a boolean vector of four
   elements.  For example, comparing \[101, 102, 103, 104\] with itself leads to

   ```python
   [True, True, True, True]
   ```

1. When we compare the first row with all the three rows, or the input matrix, we get a boolean
   matrix in the shape of 3x4.  For example, comparing the first row with all three of them leads to

   ```python
   [[True, True, True, True], [True, True, False, False], [True, True, False, True]]
   ```

1. When we compare all three rows with the input matrix, we get three 3x4 matrices.

   ```python
   [
       [[True, True, True, True], [True, True, False, False], [True, True, False, True]],
       [[True, True, False, False], [True, True, True, True], [True, True, False, False]],
       [[True, True, False, True], [True, True, False, False], [True, True, True, True]],
   ]
   ```

Using what we learned from the pairwise comparison of elements in a vector, we know that we need to
prepare the input matrix for broadcasting. For the replication of the units to be compared, which is
rows, we need to insert a dimension in front of the rows.

```python
y = b[:, None, :]

# tensor([[[101, 102, 103, 104]],
#
#         [[101, 102, 203, 204]],
#
#         [[101, 102, 303, 104]]])
```

To compare with the second and the third row, after replicated, we need to replicate the matrix. To prepare for the replication of matrix, we need to insert a dimension as the first one.

```python
x = b[None, :, :]

# tensor([[[101, 102, 103, 104],
#          [101, 102, 203, 204],
#          [101, 102, 303, 104]]])
```

Please be aware that b\[:, :\] is exactly b.

The comparison automatically broadcast x and y.

```python
print(x == y)

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

## Pairwise Compare Rows in a 3D Tensor

In practice, we have B prompts in a batch, and each prompt will have a beam of candidate tokens like
the above matrix b. The B beams form a 3D tensor R.

```python
R = torch.tensor(
    [
        [
            [91, 92, 93, 95],
            [91, 92, 94, 96],
            [91, 92, 93, 97],
        ],
        [
            [93, 94, 95, 92],
            [93, 95, 96, 93],
            [93, 94, 97, 96],
        ],
    ]
)
```

Becuase R is 3D tensor, R\[:,:,:\] is exactly R.

We treat each prompt individually and independent with other prompts.  So, for the beam b of each
prompt, we want to pairwise compare rows of it like we did in the previous section.

Using what we learned from the above two sections, we need to insert a dimension in front of the
units (rows) to be compared.

```python
y = R[:, :, None, :]
```

And we need to insert a dimension in front of the unit that is one level coarser grained than
rows, which is the matrices.

```python
x = R[:, None, :, :]
```

Then, the comparison will automatically broadcast x and y.

```python
print(x == y)

# tensor([[[[ True,  True,  True,  True],
#           [ True,  True, False, False],
#           [ True,  True,  True, False]],
#
#          [[ True,  True, False, False],
#           [ True,  True,  True,  True],
#           [ True,  True, False, False]],
#
#          [[ True,  True,  True, False],
#           [ True,  True, False, False],
#           [ True,  True,  True,  True]]],
#
#
#         [[[ True,  True,  True,  True],
#           [ True, False, False, False],
#           [ True,  True, False, False]],
#
#          [[ True, False, False, False],
#           [ True,  True,  True,  True],
#           [ True, False, False, False]],
#
#          [[ True,  True, False, False],
#           [ True, False, False, False],
#           [ True,  True,  True,  True]]]])
```

## Conclusion

### Range Indexing

- For a 1D tensor l, b\[:\]==l.
- For a 2D tensor b, b\[:,:\]==b.
- For a 3D tensor B, B\[:,:,:\]==B.

### Expand Dimension for Broadcast

For a 1D tensor, we have two options when inserting a new dimension.

- b\[None,:\], where broadcasting in the new dimension replicates the 1D tensor (vecotr),
- b\[:,None\], where broadcasting in the new dimension replicates elements.

For a 2D tensor, we have three options:

- b\[None,:,:\], where broadcasting in the new dimension replicates the 2D tensor (matrix,
- b\[:,None,:\], where broadcasting in the new dimension replicates row vectors,
- b\[:,:,None\], where broadcasting in the new dimension replicates elements.

For a 3D tensor, we have four options:

- B\[None,:,:,:\], where broadcasting in the new dimension replicates the 3D tensor,
- B\[:,None,:,:\], where broadcasting in the new dimension replicates row matrices of B,
- B\[:,:,None,:\], where broadcasting in the new dimension replicates row vectors,
- B\[:,:,:,None\], where broadcasting in the new dimension replicates elements.
