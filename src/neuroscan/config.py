"""Typed, layered configuration for every Axial Screening Assistant subsystem.

Configuration is resolved in three layers, later layers overriding earlier:

1. The field defaults declared in this module.
2. ``configs/default.yaml``.
3. Any experiment file passed to :func:`load_config` (e.g. ``configs/efficientnet_b0.yaml``),
   plus ``NEUROSCAN_*`` environment variables for deployment-time secrets.

Every consumer takes a config object rather than reading globals, which keeps
training runs reproducible and lets tests build throwaway configs in-memory.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

# Repository root: src/neuroscan/config.py -> src/neuroscan -> src -> <root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


class PathConfig(BaseModel):
    """Filesystem layout. Relative paths resolve against the repository root."""

    data_root: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    interim_dir: Path = Path("data/interim")
    processed_dir: Path = Path("data/processed")
    artifacts_dir: Path = Path("artifacts")
    runs_dir: Path = Path("artifacts/runs")
    models_dir: Path = Path("artifacts/models")
    index_dir: Path = Path("artifacts/faiss_index")
    knowledge_base_dir: Path = Path("knowledge_base")
    uploads_dir: Path = Path("instance/uploads")
    reports_dir: Path = Path("instance/reports")

    @model_validator(mode="after")
    def _absolutise(self) -> PathConfig:
        for name in type(self).model_fields:
            value = getattr(self, name)
            if not value.is_absolute():
                object.__setattr__(self, name, (PROJECT_ROOT / value).resolve())
        return self

    def ensure_all(self) -> None:
        """Create every configured directory. Safe to call repeatedly."""
        for name in type(self).model_fields:
            getattr(self, name).mkdir(parents=True, exist_ok=True)


class DatasetConfig(BaseModel):
    """Which corpus of MRI scans to train on, and how it is laid out.

    ``adapter`` selects the ingestion strategy so the clinical Grande dataset
    can replace the public contingency dataset without touching model code
    (Project Scope: "Publicly available datasets will be used as supplementary
    data if needed").
    """

    name: str = "br35h"
    adapter: Literal["imagefolder", "flat_labelled", "dicom", "manifest"] = "imagefolder"
    source_dir: Path | None = None
    task: Literal["binary", "multiclass"] = "binary"
    class_names: list[str] = Field(default_factory=lambda: ["normal", "abnormal"])

    # A regex capturing a patient/study identifier from the filename. When set,
    # splits are grouped by this key so slices from one patient never straddle
    # the train/test boundary - the single most common source of inflated
    # accuracy in small MRI studies.
    patient_id_pattern: str | None = None

    # Maps on-disk folder names onto ``class_names``. Public MRI datasets use
    # wildly inconsistent labels ("yes"/"no", "notumor"/"glioma", "healthy"),
    # and collapsing them here keeps the rest of the pipeline label-agnostic.
    # When None, a sensible default table is used - see
    # ``neuroscan.data.adapters.DEFAULT_FOLDER_CLASS_MAP``.
    folder_class_map: dict[str, str] | None = None

    supplementary: list[str] = Field(default_factory=list)

    @field_validator("class_names")
    @classmethod
    def _at_least_two_classes(cls, v: list[str]) -> list[str]:
        if len(v) < 2:
            raise ValueError("class_names must contain at least two classes")
        if len(set(v)) != len(v):
            raise ValueError("class_names must be unique")
        return v

    @model_validator(mode="after")
    def _task_matches_classes(self) -> DatasetConfig:
        if self.task == "binary" and len(self.class_names) != 2:
            raise ValueError(
                f"task='binary' requires exactly 2 class_names, got {len(self.class_names)}"
            )
        return self

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


class PreprocessingConfig(BaseModel):
    """Deterministic image standardisation applied to train, val and test alike.

    CLAHE is applied before resizing so the equalisation operates on the native
    resolution, and is confined to the luminance channel to avoid the colour
    casts that arise from equalising RGB planes independently.
    """

    image_size: int = 224
    clahe_enabled: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: int = 8
    skull_strip: bool = False  # reserved; not applied to axial JPG/PNG input
    # ImageNet statistics - required because the transfer backbones were
    # pre-trained under this normalisation.
    normalize_mean: list[float] = Field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: list[float] = Field(default_factory=lambda: [0.229, 0.224, 0.225])

    @field_validator("image_size")
    @classmethod
    def _sane_size(cls, v: int) -> int:
        if not 32 <= v <= 1024:
            raise ValueError("image_size must be between 32 and 1024")
        return v

    @field_validator("normalize_mean", "normalize_std")
    @classmethod
    def _three_channels(cls, v: list[float]) -> list[float]:
        if len(v) != 3:
            raise ValueError("normalisation statistics must have exactly 3 channels")
        return v


class AugmentationConfig(BaseModel):
    """Training-time augmentation. Applied to the training split only.

    Vertical flips are deliberately absent: an axial brain slice has a stable
    anterior/posterior orientation, and flipping it produces anatomically
    impossible images that teach the model nothing useful.
    """

    enabled: bool = True
    rotation_degrees: float = 15.0
    zoom_range: float = 0.10
    horizontal_flip: bool = True
    brightness: float = 0.20
    contrast: float = 0.20
    random_erasing_p: float = 0.0


class SplitConfig(BaseModel):
    """Train/validation/test partitioning."""

    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    stratify: bool = True
    group_by_patient: bool = True

    # Public MRI datasets contain near-duplicate images that no filename
    # reveals. Left undetected they straddle the train/test boundary and
    # inflate reported accuracy. See ``neuroscan.data.dedup``.
    #
    # The threshold is set from the measured distance distribution on Br35H,
    # not by intuition. Brain MRIs are globally similar - dark background,
    # central oval - so perceptual-hash distances between genuinely different
    # scans are already small, and a loose threshold makes union-find chain
    # the whole dataset into one cluster. Measured largest-cluster size by
    # threshold: 14 at 0, 27 at 1, then 140 at 2 and 1381 at 5. The jump at 2
    # is chaining, not duplication, so 1 is the last defensible value.
    detect_near_duplicates: bool = True
    duplicate_hamming_threshold: int = 1

    @model_validator(mode="after")
    def _ratios_sum_to_one(self) -> SplitConfig:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"split ratios must sum to 1.0, got {total:.6f}")
        if min(self.train_ratio, self.val_ratio, self.test_ratio) <= 0:
            raise ValueError("every split ratio must be strictly positive")
        return self


class TrainingConfig(BaseModel):
    """Two-stage transfer-learning schedule.

    Stage 1 trains only the new classifier head on a frozen backbone, which
    prevents large early gradients from destroying the pre-trained features.
    Stage 2 unfreezes the deepest ``unfreeze_layers`` blocks at a reduced
    learning rate.
    """

    architecture: Literal["baseline_cnn", "vgg16", "efficientnet_b0"] = "efficientnet_b0"
    pretrained: bool = True
    batch_size: int = 32
    epochs_head: int = 10
    epochs_finetune: int = 25
    lr_head: float = 1e-3
    lr_finetune: float = 1e-5
    weight_decay: float = 1e-4
    optimizer: Literal["adam", "adamw", "sgd"] = "adamw"
    scheduler: Literal["cosine", "plateau", "none"] = "cosine"
    label_smoothing: float = 0.05
    dropout: float = 0.3
    unfreeze_layers: int = 30
    class_weighting: bool = True
    early_stopping_patience: int = 8
    early_stopping_metric: Literal["val_loss", "val_auc", "val_f1"] = "val_auc"
    grad_clip_norm: float | None = 1.0
    mixed_precision: bool = True
    num_workers: int = 4
    seed: int = 42
    device: Literal["auto", "cuda", "cpu"] = "auto"

    @field_validator("batch_size", "epochs_finetune")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be a positive integer")
        return v

    @field_validator("epochs_head")
    @classmethod
    def _non_negative(cls, v: int) -> int:
        # Zero is meaningful: the scratch baseline has no pre-trained backbone
        # to freeze, so it skips stage 1 entirely.
        if v < 0:
            raise ValueError("epochs_head must be zero or a positive integer")
        return v


class EvaluationConfig(BaseModel):
    """Metrics and validation protocol (Project Design 4.1)."""

    cv_folds: int = 5
    run_cross_validation: bool = False
    # Operating threshold for the positive (abnormal) class. Deliberately below
    # 0.5: in triage, a missed abnormality costs far more than a false alarm.
    decision_threshold: float = 0.40
    tune_threshold_on_val: bool = True

    # Sensitivity floor the tuned threshold must clear on validation.
    #
    # Set high deliberately. Threshold selection otherwise maximises F1, which
    # treats a missed abnormality and a false alarm as equally costly - and in
    # triage they are not remotely equal. A false alarm costs a radiologist's
    # review; a miss costs the diagnostic delay this project exists to prevent.
    # Measured on Br35H, a 0.90 floor produced 14 missed abnormalities and zero
    # false alarms, which is the wrong side of that trade to land on.
    min_recall: float = 0.95

    # Objective the threshold search maximises above that floor. F2 weights
    # recall twice as heavily as precision; F1 weights them equally and so
    # asserts that a missed tumour and a false alarm cost the same.
    threshold_metric: Literal["f2", "f1", "youden", "balanced_accuracy"] = "f2"

    bootstrap_ci_samples: int = 1000
    target_accuracy: float = 0.90  # accuracy target


class RAGConfig(BaseModel):
    """Retrieval-Augmented Generation over the curated medical corpus."""

    # Multilingual AND asymmetric: a Nepali query must retrieve from an English
    # corpus, and a short question must match a long passage. See
    # ``neuroscan.rag.vectorstore.build_embeddings`` for why a symmetric
    # paraphrase model is the wrong tool here.
    embedding_model: str = "intfloat/multilingual-e5-base"
    embedding_device: Literal["auto", "cuda", "cpu"] = "auto"
    chunk_size: int = 900
    chunk_overlap: int = 150
    top_k: int = 5
    fetch_k: int = 20
    search_type: Literal["similarity", "mmr"] = "mmr"
    mmr_lambda: float = 0.5
    # Chunks scoring below this are dropped; if nothing survives, the advisory
    # falls back to a fixed safe response rather than inventing content.
    score_threshold: float = 0.25
    min_corpus_documents: int = 50  # corpus size requirement


class LLMConfig(BaseModel):
    """Local-first LLM provider. Configured for offline clinic deployment."""

    provider: Literal["ollama", "huggingface", "anthropic", "openai", "echo"] = "ollama"
    model: str = "llama3.1:8b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.15  # low: clinical guidance must not be creative
    max_tokens: int = 900
    timeout_seconds: int = 120
    num_ctx: int = 8192
    # Used only when provider is a hosted API; read from the environment.
    api_key_env_var: str = "NEUROSCAN_LLM_API_KEY"

    @field_validator("temperature")
    @classmethod
    def _bounded_temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("temperature must be within [0.0, 1.0]")
        return v


class ChatbotConfig(BaseModel):
    """Bilingual conversational layer."""

    default_language: Literal["en", "ne"] = "en"
    supported_languages: list[str] = Field(default_factory=lambda: ["en", "ne"])
    max_history_turns: int = 8
    max_question_chars: int = 1000


class WebConfig(BaseModel):
    """Flask application settings."""

    host: str = "127.0.0.1"
    port: int = 5000
    debug: bool = False
    max_upload_mb: int = 16
    allowed_extensions: list[str] = Field(default_factory=lambda: ["jpg", "jpeg", "png"])
    secret_key_env_var: str = "NEUROSCAN_SECRET_KEY"
    session_lifetime_minutes: int = 60
    # Uploaded scans are purged after this many hours (docs/ETHICS.md).
    retain_uploads_hours: int = 24


class Config(BaseModel):
    """Root configuration object passed through the whole system."""

    project_name: str = "Axial Screening Assistant"
    paths: PathConfig = Field(default_factory=PathConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    chatbot: ChatbotConfig = Field(default_factory=ChatbotConfig)
    web: WebConfig = Field(default_factory=WebConfig)

    def dump_yaml(self, path: Path) -> None:
        """Persist the fully-resolved config beside a run's artefacts.

        Every training run writes this file so any reported number can be
        traced back to the exact settings that produced it.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.model_dump(mode="json")
        path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), "utf-8")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` without mutating either."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    data = yaml.safe_load(path.read_text("utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration file must contain a mapping at the top level: {path}")
    return data


# Environment overrides for values that must not be committed or that change
# per deployment. Mapped as NEUROSCAN_<VAR> -> dotted config path.
_ENV_OVERRIDES: dict[str, tuple[str, type]] = {
    "NEUROSCAN_LLM_PROVIDER": ("llm.provider", str),
    "NEUROSCAN_LLM_MODEL": ("llm.model", str),
    "NEUROSCAN_LLM_BASE_URL": ("llm.base_url", str),
    "NEUROSCAN_DEVICE": ("training.device", str),
    "NEUROSCAN_WEB_PORT": ("web.port", int),
    "NEUROSCAN_WEB_DEBUG": ("web.debug", bool),
    "NEUROSCAN_DATA_ROOT": ("paths.data_root", str),
}


def _apply_env_overrides(payload: dict[str, Any]) -> dict[str, Any]:
    for env_var, (dotted, caster) in _ENV_OVERRIDES.items():
        raw = os.environ.get(env_var)
        if raw is None or raw == "":
            continue
        # bool("false") is True, so booleans need explicit parsing rather than
        # the caster.
        value: Any = (
            raw.strip().lower() in {"1", "true", "yes", "on"}
            if caster is bool
            else caster(raw)
        )
        section, _, leaf = dotted.partition(".")
        payload.setdefault(section, {})[leaf] = value
    return payload


def load_config(
    experiment: str | Path | None = None,
    *,
    overrides: dict[str, Any] | None = None,
    use_env: bool = True,
) -> Config:
    """Build the resolved :class:`Config`.

    Args:
        experiment: Optional experiment YAML. May be a path, or a bare name
            resolved against ``configs/`` (``"vgg16"`` -> ``configs/vgg16.yaml``).
        overrides: Nested dict merged last, above files and environment. Used
            by CLI flags and tests.
        use_env: Apply ``NEUROSCAN_*`` environment overrides.

    Raises:
        FileNotFoundError: If a named config file does not exist.
        pydantic.ValidationError: If the merged result violates any constraint.
    """
    payload: dict[str, Any] = {}

    if DEFAULT_CONFIG_PATH.exists():
        payload = _read_yaml(DEFAULT_CONFIG_PATH)

    if experiment is not None:
        exp_path = Path(experiment)
        if not exp_path.suffix:
            exp_path = PROJECT_ROOT / "configs" / f"{exp_path.name}.yaml"
        elif not exp_path.is_absolute():
            exp_path = (PROJECT_ROOT / exp_path).resolve()
        payload = _deep_merge(payload, _read_yaml(exp_path))

    if use_env:
        payload = _apply_env_overrides(payload)

    if overrides:
        payload = _deep_merge(payload, overrides)

    return Config(**payload)


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "PROJECT_ROOT",
    "AugmentationConfig",
    "ChatbotConfig",
    "Config",
    "DatasetConfig",
    "EvaluationConfig",
    "LLMConfig",
    "PathConfig",
    "PreprocessingConfig",
    "RAGConfig",
    "SplitConfig",
    "TrainingConfig",
    "WebConfig",
    "load_config",
]
