# Speculative Sampling

Text decoding involves sampling tokens from the LLM, represented by v ~ P(v), in response to a query.  Unfortunately, this sampling technique, known as auto-regression, is computationally intensive. We sample from the drafter, denoted by v~Q(v), and use an algorithm to accept or reject v, making accepted v's look like they were sampled from P(v).

When we sample tokens from the drafter, the beam search method returns log Q(v), in addition to v.  To verify v, we pass the token before v to the LLM, which returns log P(v).  Given v, log Q(v), and log P(v), what is the best algorithm for deciding whether to keep or reject v?

To have a general understanding of the challenge, consider a simple vocabulary with only two tokens: A and B.  Assume that Q(A)=Q(B)=1/2, P(A)=1/4, and P(B)=3/4.

```
                              P(B)=3/4
                              ▄▄
Q(A)=1/2             Q(B)=1/2 ██
     ┌─┐ P(A)=1/4         ┌─┐ ██
     │ │ ▄▄               │ │ ██
     └─┘ ▀▀               └─┘ ▀▀
----------------------------------------
      A                     B
```

Assume that sampling from Q(v) yields a token A. Because the real probability of A, P(A), is lower than Q(A), we should keep it with probability P(A)/Q(A), or reject it with probability 1-P(A)/Q(A).  Assume that sampling from Q(v) yielded B. Because the real probability P(B) is greater than Q(B), we wish to keep the token B.

These intuitions seem similar to a sampling approach known as rejection sampling. However, they are not. Speculative sampling is not iterative but rejection sampling is. Also, unlike in other rejection sampling applications that use continuous probability density functions, the distribution over vocabulary is discrete. This makes things easier because we don't have to choose a normalizing constant c such that c Q(v) > P(v) for all v, as is required in the continuous situations.

Append A.1 of the paper at https://arxiv.org/pdf/2211.17192 shows that the above intuition is exactly what we need to do to alter drafter samples such that they look like LLM samples.
