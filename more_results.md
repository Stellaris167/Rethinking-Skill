# More Results

## Entropy Analysis

Wrong-answer entropy (bits) per question in the pass@8 sampling runs.

| Model | DAPO Wrong-Answer Entropy &darr; |
|---|---:|
| Qwen3.5-2B | 1.230 |
| Qwen3.5-4B | 1.248 |
| Qwen3.5-9B | 0.968 |
| Qwen3.5-27B | 0.475 |

Smaller models show higher wrong-answer entropy across repeated samples: they produce a wider range of incorrect answers to the same question. We hypothesize that this variability makes skill optimization more difficult. When errors vary from one sample to another, the optimizer has less consistent evidence about what the skill should address. A rule that helps with one sampled failure may do little for the next, making it harder to find a stable direction for optimization and achieve consistent performance gains through skills.
