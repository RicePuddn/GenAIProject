# Experiment configs

One YAML file per experiment (E1, E2, ...). Each one starts with `extends: base.yaml` and lists only
what it changes, so it is obvious what each experiment varies. Every run writes its fully
resolved config next to its results.

Planned experiments, to be confirmed by the team:
- architecture comparison (S1-S4)
- retrieval comparison (cosine, hybrid RRF, hybrid + reranking) on one fixed PDF corpus
- S4 supervisor strategy (ReAct vs plan-and-execute)
- ablations (remove one component at a time)
- failure injection (tool timeouts, duplicate requests, policy change mid-run)
- robustness (typos, paraphrases)
