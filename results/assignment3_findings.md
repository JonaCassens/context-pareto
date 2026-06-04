# Assignment 3 Findings

## Token-Quality Tradeoff Summary

      strategy                               hyperparams_key  runs  accuracy_pct  avg_token_reduction_pct
        hybrid k=6, recent_window=2, summary_token_limit=160    36     63.888889                -3.773306
        hybrid k=4, recent_window=2, summary_token_limit=160    36     52.777778                10.158995
        hybrid k=2, recent_window=2, summary_token_limit=160    36     58.333333                35.127778
sliding_window                                      window=6    36     16.666667                21.103112
sliding_window                                      window=4    36      0.000000                45.948371
sliding_window                                      window=2    36      0.000000                73.744851
 summarization                               token_limit=400    36     61.111111                 0.000000
 summarization                               token_limit=200    36     50.000000                19.429003
 summarization                               token_limit=100    36     30.555556                57.739381

## Cliff Points
- No cliff detected by the configured rule (accuracy drop > token gain between adjacent compression levels).

## Recommended Strategy

Recommended strategy: hybrid

Rationale: highest mean accuracy across its hyperparameter settings, with token reduction used as tie-breaker.

Strategy rollup:
      strategy  accuracy_pct  avg_token_reduction_pct
        hybrid     58.333333                13.837822
 summarization     47.222222                25.722795
sliding_window      5.555556                46.932111

## One Concrete Failure Case

```text
Conversation ID: 1e7c9f6d-4a1b-4e0f-9c2d-8b3a5f7e1d0c
Strategy: hybrid (k=2, recent_window=2, summary_token_limit=160)
Token reduction ratio: 0.582
Ground truth: $91,800.00
Model answer: I cannot provide the standard unit price for
Critical turn indices: [0, 8]
Kept turn indices: [0, 1, 7, 8]
Missing critical indices: []
Mechanism: Compression retained critical indices but likely lost detail fidelity during summarization or semantic selection.
Judge reasoning: Candidate answer does not contain a parseable numeric value.
```