# Rationale for gating

Wang et al. organize inference around prediction, selection, execution, and updating. System 1 estimates candidate interactions' value before invocation. System 2 executes selected interactions, and their outcomes inform the state and later decision models. The paper discusses screening cost, missed useful actions, and missing outcome labels for skipped interactions. [Wang et al., 2026, Sections 6.3 and 7.2](https://arxiv.org/abs/2610.39497)

In DRIFT-inspired BM25 RAG, the Jev sufficiency gate decides whether to return the current draft and end retrieval. The necessity gate decides whether to execute a specific follow-up. Qwen generates follow-ups, reads retrieved evidence, and updates the draft.

The experiments gate on Jev's confidence over the available options. Compact inputs use the same temperature scaling. The threshold controls which actions execute, and matched end-to-end results measure the actual benefit. The cost ledger includes Jev input and decision time alongside Qwen input, generation, and call time.
