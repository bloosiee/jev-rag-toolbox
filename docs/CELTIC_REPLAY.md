# Real error diagnosis and replay with Celtic

HotpotQA question `5abe48675542991f66106119` asks which club Charlie Nicholas played for in the 1982 Scottish League Cup final. The reference answer is **Celtic**.

## Diagnosis

The v1 and v2 pipelines shared the initial draft `Charlie Nicholas` and the same initial documents. The first sufficiency decision had 82.23% confidence with v1, below 95%, so v1 continued. With v2, confidence reached 95.30% and the pipeline submitted the player's name. The v1 pipeline then executed a follow-up distinguishing Celtic from Rangers and returned the club.

Inspect the recorded v2 decision.

```bash
jev rag trace --input results/hotpotqa400_v2_traces.jsonl.gz \
  --id 5abe48675542991f66106119 --step 1
```

## Model replay

The replay used the original Qwen3.5-4B generator and Jev v2, reusing the draft and its two retrieved documents. It forced each option at the first sufficiency node and then ran the same pipeline.

| Forced option | Final answer | EM | Post-fork Qwen calls | Post-fork Jev calls | Post-fork total tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| `sufficient` | Charlie Nicholas | 0 | 0 | 0 | 0 |
| `continue` | Celtic | 1 | 3 | 2 | 2,405 |

Continuing executed one follow-up, added one document, and rewrote the answer to `Celtic`. The next sufficiency decision stopped at 99.80% confidence. The replay took about 8.52 additional seconds in this run, excluding the shared prefix and model loading.

These figures come from a real model replay. The repository's offline demo uses deterministic adapters. With your own full trace, the same command can select any recorded Jev node. [Replay commands and input format](EXPERIMENTS.md)
