## Deep Dive

1. [The Recurrent Drafter Model](drafter.md) delves into the architecture of the drafter model and the beam search algorithm that proposes candidate token sequences.

1. [Speculative Sampling](speculative_sampling.md) discusses the [speculative sampling](https://arxiv.org/abs/2302.01318) algorithm, which accepts or rejects proposed candidate tokens.

1. [Tree Attention](tree_attention.md) outlines how to organize proposed candidate token sequences into a prefix tree, thereby saving FLOPs in the acceptance or rejection of tokens. It's important to note that recurrent drafting differs from Medusa and other methods by not using a fixed-structure prefix tree. Instead, this algorithm calculates the optimal tree structure based on the proposed token sequences, taking into account the temporal dependencies between tokens.

1. [Convert a Beam Into a Prefix Tree](beam_to_prefix_tree.md) is a critical part of building the tree attention given a beam of candidate tokens generated by the drafter model.

1. [Pairwise Comparison of Low-dimensional Embeddings In A Tensor](pairwise_comparison.md) is a foundation of [Convert a Beam Into a Prefix Tree](beam_to_prefix_tree.md).

## Benchmarking

1. [Comparing With Auto-regressive Decoding](parity_check.md) details the greedy search mode of recurrent drafting, which bypasses speculative sampling. In this mode, the outputs are identical to those from auto-regressive decoding, provided there is sufficient floating-point precision.

1. [Screen Recording the Performance](record_screen.md) describes how to simultaneously run auto-regressive decoding and recurrent drafting algorithms while recording their performance. This technique offers an intuitive demonstration of the performance differences.

1. [Performance w.r.t. Beam Search](recurrent_drafting/benchmark/perf_wrt_candidates) explains how to conduct parallel inference jobs to comprehensively explore the optimal beam search parameters for achieving the best performance on specific types of GPUs and data types. The results are presented in an animation that vividly illustrates how to maximize the utilization of a particular GPU model.
