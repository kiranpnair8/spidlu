"""Configuration loading for Phase 1 RQ1."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Phase1Config:
    model_name_or_path: str
    model_revision: str | None = None
    variants: list[str] = field(default_factory=lambda: [
        "ann_original",
        "spidlu",
        "ann_compute_matched",
        "quantized_activation",
    ])
    dataset_name: str = "Salesforce/wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    text_column: str = "text"
    max_seq_len: int = 512
    train_split: str = "train"
    eval_split: str = "validation"
    downstream_split: str = "validation"
    batch_size: int = 1
    eval_batch_size: int = 1
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    max_train_steps: int = 100
    max_train_tokens: int | None = None
    eval_every_steps: int = 50
    save_every_steps: int = 50
    seed: int = 42
    spidlu_alpha: float = 0.9
    spidlu_threshold: float = 1.0
    spidlu_T: int = 4
    quantized_levels: int | None = None
    generation_prompts: list[str] = field(default_factory=lambda: [
        "The future of language models is",
        "In a careful scientific experiment,",
    ])
    generation_max_new_tokens: int = 32
    output_dir: str = "models/phase1_rq1"
    smoke: bool = False


def load_config(path):
    data = yaml.safe_load(Path(path).read_text()) or {}
    flat = {}
    for section in ("model", "data", "training", "spidlu", "generation", "output"):
        flat.update(data.get(section, {}) or {})
    if "name_or_path" in flat:
        flat["model_name_or_path"] = flat.pop("name_or_path")
    if "revision" in flat:
        flat["model_revision"] = flat.pop("revision")
    if "T_steps" in flat:
        flat["spidlu_T"] = flat.pop("T_steps")
    if "alpha" in flat:
        flat["spidlu_alpha"] = flat.pop("alpha")
    if "threshold" in flat:
        flat["spidlu_threshold"] = flat.pop("threshold")
    allowed = set(Phase1Config.__dataclass_fields__)
    return Phase1Config(**{k: v for k, v in flat.items() if k in allowed})
