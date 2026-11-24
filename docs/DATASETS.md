# Data card

| Dataset | Train questions / states | Validation questions / states | Test questions / states |
| --- | ---: | ---: | ---: |
| SufficiencyBench v1.1 | 1,352 / 2,704 | 264 / 528 | 1,069 / 2,138 |
| RAG-Necessity-987 | 697 / 1,822 | 140 / 370 | 150 / 396 |
| Trajectory sufficiency (v2) | 389 / 927 | 77 / 189 | 82 / 196 |

## SufficiencyBench v1.1

We use the official split and sufficiency labels. Each question can have multiple retrieved contexts, and evaluation scores each state. The task is to judge whether the visible documents can fully answer the question. The upstream text is subject to its source terms. [SufficiencyBench](https://github.com/inab-lab/SufficiencyBench)

## RAG-Necessity-987

This dataset uses retrieval trajectories from HotpotQA questions. GLM and DeepSeek teachers label follow-ups as worth executing, already covered by evidence, or unhelpful to the original question. Uncertain labels are excluded. We used the first 987 completed questions and grouped states by question with seed 42. Label availability may affect the sample composition. Evaluation measures agreement with the teacher labels.

## Trajectory sufficiency (v2)

Samples come from ungated iterative RAG on the HotpotQA training split. Inputs contain the current draft, retrieved evidence, and follow-up history. A draft is labeled ready to submit when it matches the reference answer and the retrieved text contains all annotated supporting sentences. Explicit abstentions, reversed judgments, incorrect integers, and individually reviewed wrong or missing answers are labeled continue. Cases with possible aliases or alternative evidence are held out of this labeling process. Both tasks and all rounds for a question use the same split. The HotpotQA 400 pipeline benchmark uses separate questions.

## File format

Each `train.json` row includes `id`, `task_id`, `state`, the fixed options, and a label. Model input comprises the state and fixed judgment question. The label is used in the loss. Validation and test use `requests.json` for inputs and `gold.json` for labels keyed by ID. The three splits are disjoint by question ID within each task. Version control excludes dataset contents by default. Review source and redistribution terms before publishing them.
