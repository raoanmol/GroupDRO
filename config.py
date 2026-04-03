import yaml
from typing import Optional, List
from dataclasses import dataclass, fields
from types import SimpleNamespace


@dataclass
class DataConfig:
    dataset: str
    shift_type: str
    target_name: Optional[str] = None
    confounder_names: Optional[List[str]] = None
    root_dir: Optional[str] = None
    fraction: float = 1.0
    val_fraction: float = 0.1
    num_val_samples_per_class: Optional[int] = None
    batch_size: int = 32
    num_workers: int = 4
    reweight_groups: bool = False
    augment_data: bool = False
    minority_fraction: Optional[float] = None
    imbalance_ratio: Optional[float] = None


@dataclass
class ModelConfig:
    name: str = "resnet50"
    train_from_scratch: bool = False


@dataclass
class TrainingConfig:
    n_epochs: int = 4
    lr: float = 0.001
    weight_decay: float = 5e-5
    scheduler: bool = False
    gamma: float = 0.1


@dataclass
class RobustnessConfig:
    robust: bool = False
    robust_step_size: float = 0.01
    alpha: float = 0.2
    generalization_adjustment: str = "0.0"
    automatic_adjustment: bool = False
    use_normalized_loss: bool = False
    btl: bool = False
    minimum_variational_weight: float = 0.0
    hinge: bool = False


@dataclass
class LoggingConfig:
    log_dir: str = "./logs"
    log_every: int = 50
    save_step: int = 10
    save_best: bool = False
    save_last: bool = False
    show_progress: bool = False
    resume: bool = False


@dataclass
class Config:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    robustness: RobustnessConfig
    logging: LoggingConfig
    seed: int = 0
    device: str = "auto"

    def to_namespace(self) -> SimpleNamespace:
        """Flatten nested Config into a flat SimpleNamespace compatible with
        the legacy args interface used by train.py, data/*.py, and loss.py."""
        flat = {}

        flat['seed'] = self.seed
        flat['device'] = self.device

        for f in fields(self.data):
            flat[f.name] = getattr(self.data, f.name)

        # model.name -> args.model (downstream code expects args.model)
        flat['model'] = self.model.name
        flat['train_from_scratch'] = self.model.train_from_scratch

        for f in fields(self.training):
            flat[f.name] = getattr(self.training, f.name)

        for f in fields(self.robustness):
            flat[f.name] = getattr(self.robustness, f.name)

        for f in fields(self.logging):
            flat[f.name] = getattr(self.logging, f.name)

        return SimpleNamespace(**flat)


def load_config(yaml_path: str) -> Config:
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f)

    data = DataConfig(**raw["data"])
    model = ModelConfig(**raw.get("model", {}))
    training = TrainingConfig(**raw.get("training", {}))
    robustness = RobustnessConfig(**raw.get("robustness", {}))
    logging_cfg = LoggingConfig(**raw.get("logging", {}))

    return Config(
        data=data,
        model=model,
        training=training,
        robustness=robustness,
        logging=logging_cfg,
        seed=raw.get("seed", 0),
        device=raw.get("device", "auto"),
    )


def check_config(config: Config) -> None:
    from data.data import dataset_attributes, shift_types
    from models import model_attributes

    if config.data.dataset not in dataset_attributes:
        valid = ', '.join(dataset_attributes.keys())
        raise ValueError(f"Unknown dataset '{config.data.dataset}'. Valid: {valid}")

    if config.model.name not in model_attributes:
        valid = ', '.join(model_attributes.keys())
        raise ValueError(f"Unknown model '{config.model.name}'. Valid: {valid}")

    if config.data.shift_type not in shift_types:
        raise ValueError(f"Unknown shift_type '{config.data.shift_type}'. Valid: {shift_types}")

    if config.data.shift_type == 'confounder':
        if not config.data.confounder_names:
            raise ValueError("confounder_names required when shift_type='confounder'")
        if not config.data.target_name:
            raise ValueError("target_name required when shift_type='confounder'")

    if config.data.shift_type.startswith('label_shift'):
        if config.data.minority_fraction is None:
            raise ValueError("minority_fraction required for label_shift")
        if config.data.imbalance_ratio is None:
            raise ValueError("imbalance_ratio required for label_shift")
