"""Experimental multimodal features and model comparison (branch experiment/multimodal).

Isolated from the production pipeline: reads Stage 3/4 artefacts, writes only under data/processed/multimodal and
outputs/multimodal, and never touches Stage 4/5 files, the frozen selection record or the default inspection policy.
"""
