import os, shutil
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from config import load_config, check_config
from models import model_attributes, MODEL_REGISTRY, create_model
from data.data import dataset_attributes, shift_types, prepare_data, log_data
from utils import set_seed, Logger, log_args
from train import train
from compute_tracker import ComputeTracker


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


def subset_loader(loader: DataLoader, num_samples: int) -> DataLoader:
    indices = list(range(min(num_samples, len(loader.dataset))))
    subset = Subset(loader.dataset, indices)
    return DataLoader(
        subset,
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )


def main():
    parser = argparse.ArgumentParser(description="GroupDRO Training")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    parser.add_argument(
        "--seed", type=int, default=None, help="Override seed from config"
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Quick test run with subset data and 2 epochs",
    )
    cli_args = parser.parse_args()

    # Load and validate config
    config = load_config(cli_args.config)

    if cli_args.seed is not None:
        config.seed = cli_args.seed

    check_config(config)

    # Convert to flat namespace for backward compatibility with train.py, data/*.py, loss.py
    args = config.to_namespace()

    # BERT-specific configs (programmatic, not in YAML)
    if args.model == "bert":
        args.max_grad_norm = 1.0
        args.adam_epsilon = 1e-8
        args.warmup_steps = 0

    if os.path.exists(args.log_dir) and args.resume:
        resume = True
        mode = "a"
    else:
        resume = False
        mode = "w"

    ## Initialize logs
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    logger = Logger(os.path.join(args.log_dir, "log.txt"), mode)
    # Record args
    log_args(args, logger)

    device = resolve_device(config.device)
    logger.write(f"Using device: {device}\n")

    set_seed(args.seed)

    # Data
    # Test data for label_shift_step is not implemented yet
    test_data = None
    test_loader = None
    id_val_data = None
    ood_val_data = None
    val_data = None
    if args.shift_type == "confounder":
        splits = prepare_data(args, train=True)
        if len(splits) == 4:
            train_data, id_val_data, ood_val_data, test_data = splits
        else:
            train_data, val_data, test_data = splits
    elif args.shift_type == "label_shift_step":
        train_data, val_data = prepare_data(args, train=True)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": True,
    }
    train_loader = train_data.get_loader(
        train=True, reweight_groups=args.reweight_groups, **loader_kwargs
    )
    if test_data is not None:
        test_loader = test_data.get_loader(
            train=False, reweight_groups=None, **loader_kwargs
        )

    data = {}
    data["train_loader"] = train_loader
    data["test_loader"] = test_loader
    data["train_data"] = train_data
    data["test_data"] = test_data

    if id_val_data is not None:
        data["id_val_data"] = id_val_data
        data["id_val_loader"] = id_val_data.get_loader(
            train=False, reweight_groups=None, **loader_kwargs
        )
        data["ood_val_data"] = ood_val_data
        data["ood_val_loader"] = ood_val_data.get_loader(
            train=False, reweight_groups=None, **loader_kwargs
        )
        data["val_data"] = None
        data["val_loader"] = None
    else:
        val_loader = val_data.get_loader(
            train=False, reweight_groups=None, **loader_kwargs
        )
        data["val_data"] = val_data
        data["val_loader"] = val_loader
        data["id_val_data"] = None
        data["ood_val_data"] = None

    # Test mode: subset all loaders and override epochs
    if cli_args.test_mode:
        print("[TEST MODE] Running with subset data and 2 epochs")
        data["train_loader"] = subset_loader(data["train_loader"], 20)
        if data["test_loader"] is not None:
            data["test_loader"] = subset_loader(data["test_loader"], 5)
        if data.get("val_loader") is not None:
            data["val_loader"] = subset_loader(data["val_loader"], 5)
        if data.get("id_val_loader") is not None:
            data["id_val_loader"] = subset_loader(data["id_val_loader"], 5)
        if data.get("ood_val_loader") is not None:
            data["ood_val_loader"] = subset_loader(data["ood_val_loader"], 5)
        args.n_epochs = 2

    n_classes = train_data.n_classes

    log_data(data, logger)

    ## Initialize model
    pretrained = not args.train_from_scratch
    if model_attributes[args.model]["feature_type"] in ("precomputed", "raw_flattened"):
        assert pretrained
        # Load precomputed features
        d = train_data.input_size()[0]
        model = nn.Linear(d, n_classes)
        model.has_aux_logits = False
    elif args.model in MODEL_REGISTRY:
        model = create_model(args.model, n_classes, pretrained=pretrained)
    elif args.model == "bert":
        assert args.dataset == "MultiNLI"

        from pytorch_transformers import BertConfig, BertForSequenceClassification

        config_class = BertConfig
        model_class = BertForSequenceClassification

        bert_config = config_class.from_pretrained(
            "bert-base-uncased", num_labels=3, finetuning_task="mnli"
        )
        model = model_class.from_pretrained(
            "bert-base-uncased", from_tf=False, config=bert_config
        )
    else:
        raise ValueError("Model not recognized.")

    if resume:
        model.load_state_dict(
            torch.load(
                os.path.join(args.log_dir, "last_model.pth"), map_location=device
            )
        )

    logger.flush()

    ## Define the objective
    if args.hinge:
        assert args.dataset in ["CelebA", "CUB"]  # Only supports binary

        def hinge_loss(yhat, y):
            # The torch loss takes in three arguments so we need to split yhat
            # It also expects classes in {+1.0, -1.0} whereas by default we give them in {0, 1}
            # Furthermore, if y = 1 it expects the first input to be higher instead of the second,
            # so we need to swap yhat[:, 0] and yhat[:, 1]...
            torch_loss = torch.nn.MarginRankingLoss(margin=1.0, reduction="none")
            y = (y.float() * 2.0) - 1.0
            return torch_loss(yhat[:, 1], yhat[:, 0], y)

        criterion = hinge_loss
    else:
        criterion = torch.nn.CrossEntropyLoss(reduction="none")

    if resume:
        # Read epoch offset from metrics.csv if resuming
        metrics_path = os.path.join(args.log_dir, "metrics.csv")
        if os.path.exists(metrics_path):
            import csv
            with open(metrics_path) as f:
                rows = list(csv.DictReader(f))
            epoch_offset = int(rows[-1]["epoch"]) + 1 if rows else 0
            logger.write(f"starting from epoch {epoch_offset}")
        else:
            epoch_offset = 0
    else:
        epoch_offset = 0

    compute_tracker = (
        ComputeTracker(args.log_dir, device) if args.track_compute else None
    )

    try:
        train(
            model,
            criterion,
            data,
            logger,
            args,
            epoch_offset=epoch_offset,
            compute_tracker=compute_tracker,
        )
    finally:
        if compute_tracker is not None:
            compute_tracker.save()
            compute_tracker.close()

    # Test mode cleanup
    if cli_args.test_mode:
        print("[TEST MODE] Run successful. Cleaning up test artifacts...")
        shutil.rmtree(args.log_dir)
        print(f"[TEST MODE] Removed {args.log_dir}")

    print("Done.")


if __name__ == "__main__":
    main()
