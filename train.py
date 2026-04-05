import os
import json
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

import numpy as np
from tqdm import tqdm

from utils import AverageMeter, accuracy, get_device
from loss import LossComputer

try:
    from pytorch_transformers import AdamW, WarmupLinearSchedule
except (ModuleNotFoundError, ImportError):
    try:
        from torch.optim import AdamW
        from transformers import (
            get_linear_schedule_with_warmup as WarmupLinearSchedule,
        )
    except (ModuleNotFoundError, ImportError):
        pass  # Only needed for BERT models


def run_epoch(
    epoch,
    model,
    optimizer,
    loader,
    loss_computer,
    logger,
    csv_logger,
    args,
    is_training,
    show_progress=False,
    log_every=50,
    scheduler=None,
):
    """
    scheduler is only used inside this function if model is bert.
    """

    if is_training:
        model.train()
        if args.model == "bert":
            model.zero_grad()
    else:
        model.eval()

    if show_progress:
        prog_bar_loader = tqdm(loader)
    else:
        prog_bar_loader = loader

    # Epoch-level accumulators (independent of loss_computer resets)
    epoch_loss_sum = 0.0
    epoch_correct = 0
    epoch_total = 0

    with torch.set_grad_enabled(is_training):
        for batch_idx, batch in enumerate(prog_bar_loader):
            batch = tuple(t.to(get_device()) for t in batch)
            x = batch[0]
            y = batch[1]
            g = batch[2]
            if args.model == "bert":
                input_ids = x[:, :, 0]
                input_masks = x[:, :, 1]
                segment_ids = x[:, :, 2]
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=input_masks,
                    token_type_ids=segment_ids,
                    labels=y,
                )[1]  # [1] returns logits
            else:
                outputs = model(x)

            loss_main = loss_computer.loss(outputs, y, g, is_training)

            # Track epoch-level stats
            epoch_loss_sum += loss_main.item() * y.size(0)
            epoch_correct += (torch.argmax(outputs, 1) == y).sum().item()
            epoch_total += y.size(0)

            if is_training:
                if args.model == "bert":
                    loss_main.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )
                    scheduler.step()
                    optimizer.step()
                    model.zero_grad()
                else:
                    optimizer.zero_grad()
                    loss_main.backward()
                    optimizer.step()

            if is_training and (batch_idx + 1) % log_every == 0:
                if csv_logger is not None:
                    csv_logger.log(epoch, batch_idx, loss_computer.get_stats(model, args))
                    csv_logger.flush()
                loss_computer.log_stats(logger, is_training)
                loss_computer.reset_stats()

        if (not is_training) or loss_computer.batch_count > 0:
            if csv_logger is not None:
                csv_logger.log(epoch, batch_idx, loss_computer.get_stats(model, args))
                csv_logger.flush()
            loss_computer.log_stats(logger, is_training)
            if is_training:
                loss_computer.reset_stats()

    return {
        "avg_loss": epoch_loss_sum / max(epoch_total, 1),
        "avg_acc": epoch_correct / max(epoch_total, 1),
    }


def log_predictions_json(model, dro_dataset, output_path, args):
    """Evaluate model on a DRODataset and write per-sample predictions to JSON.

    Each entry contains file_path, ground_truth, predicted_class, and logits.
    """
    model.eval()
    device = get_device()

    # Resolve the underlying full dataset and index mapping
    inner = dro_dataset.dataset
    if isinstance(inner, Subset):
        full_dataset = inner.dataset
        indices = inner.indices
    else:
        full_dataset = inner
        indices = list(range(len(inner)))

    has_file_paths = hasattr(full_dataset, "data_dir") and hasattr(
        full_dataset, "filename_array"
    )

    loader = DataLoader(
        dro_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    predictions = []
    sample_idx = 0

    with torch.no_grad():
        for batch in loader:
            batch = tuple(t.to(device) for t in batch)
            x, y, g = batch[0], batch[1], batch[2]

            if args.model == "bert":
                input_ids = x[:, :, 0]
                input_masks = x[:, :, 1]
                segment_ids = x[:, :, 2]
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=input_masks,
                    token_type_ids=segment_ids,
                    labels=y,
                )[1]
            else:
                outputs = model(x)

            for i in range(y.size(0)):
                orig_idx = indices[sample_idx]

                if has_file_paths:
                    file_path = os.path.join(
                        full_dataset.data_dir, full_dataset.filename_array[orig_idx]
                    )
                else:
                    file_path = str(orig_idx)

                predictions.append(
                    {
                        "file_path": file_path,
                        "ground_truth": y[i].item(),
                        "predicted_class": torch.argmax(outputs[i]).item(),
                        "logits": [round(v, 4) for v in outputs[i].cpu().tolist()],
                    }
                )
                sample_idx += 1

    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=2)


def train(
    model,
    criterion,
    dataset,
    logger,
    args,
    epoch_offset,
    compute_tracker=None,
):
    model = model.to(get_device())

    # process generalization adjustment stuff
    adjustments = [float(c) for c in args.generalization_adjustment.split(",")]
    assert len(adjustments) in (1, dataset["train_data"].n_groups)
    if len(adjustments) == 1:
        adjustments = np.array(adjustments * dataset["train_data"].n_groups)
    else:
        adjustments = np.array(adjustments)

    train_loss_computer = LossComputer(
        criterion,
        is_robust=args.robust,
        dataset=dataset["train_data"],
        alpha=args.alpha,
        gamma=args.gamma,
        adj=adjustments,
        step_size=args.robust_step_size,
        normalize_loss=args.use_normalized_loss,
        btl=args.btl,
        min_var_weight=args.minimum_variational_weight,
    )

    # BERT uses its own scheduler and optimizer
    if args.model == "bert":
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon
        )
        t_total = len(dataset["train_loader"]) * args.n_epochs
        print(f"\nt_total is {t_total}\n")
        scheduler = WarmupLinearSchedule(
            optimizer, warmup_steps=args.warmup_steps, t_total=t_total
        )
    else:
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
        )
        if args.scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                "min",
                factor=0.1,
                patience=5,
                threshold=0.0001,
                min_lr=0,
                eps=1e-08,
            )
        else:
            scheduler = None

    use_4way = dataset.get("id_val_data") is not None

    # Open metrics.csv for epoch-level logging
    metrics_path = os.path.join(args.log_dir, "metrics.csv")
    metrics_file = open(metrics_path, "w")
    if use_4way:
        metrics_file.write("epoch,train_loss,train_acc,id_val_acc,ood_val_acc\n")
    else:
        metrics_file.write("epoch,train_loss,train_acc,val_acc\n")
    metrics_file.flush()

    best_val_acc = 0
    best_id_val_acc = 0
    best_ood_val_acc = 0

    if compute_tracker is not None:
        compute_tracker.start_training()

    for epoch in range(epoch_offset, epoch_offset + args.n_epochs):
        logger.write("\nEpoch [%d]:\n" % epoch)
        logger.write("Training:\n")
        if compute_tracker is not None:
            compute_tracker.start_phase(
                epoch,
                "train",
                num_samples=len(dataset["train_loader"].dataset),
                num_batches=len(dataset["train_loader"]),
            )
        train_epoch_stats = run_epoch(
            epoch,
            model,
            optimizer,
            dataset["train_loader"],
            train_loss_computer,
            logger,
            None,
            args,
            is_training=True,
            show_progress=args.show_progress,
            log_every=args.log_every,
            scheduler=scheduler,
        )
        if compute_tracker is not None:
            compute_tracker.end_phase(epoch, "train")

        if use_4way:
            logger.write("\nID Validation:\n")
            id_val_loss_computer = LossComputer(
                criterion,
                is_robust=args.robust,
                dataset=dataset["id_val_data"],
                step_size=args.robust_step_size,
                alpha=args.alpha,
            )
            if compute_tracker is not None:
                compute_tracker.start_phase(
                    epoch,
                    "id_val",
                    num_samples=len(dataset["id_val_loader"].dataset),
                    num_batches=len(dataset["id_val_loader"]),
                )
            run_epoch(
                epoch,
                model,
                optimizer,
                dataset["id_val_loader"],
                id_val_loss_computer,
                logger,
                None,
                args,
                is_training=False,
            )
            if compute_tracker is not None:
                compute_tracker.end_phase(epoch, "id_val")

            logger.write("\nOOD Validation:\n")
            ood_val_loss_computer = LossComputer(
                criterion,
                is_robust=args.robust,
                dataset=dataset["ood_val_data"],
                step_size=args.robust_step_size,
                alpha=args.alpha,
            )
            if compute_tracker is not None:
                compute_tracker.start_phase(
                    epoch,
                    "ood_val",
                    num_samples=len(dataset["ood_val_loader"].dataset),
                    num_batches=len(dataset["ood_val_loader"]),
                )
            run_epoch(
                epoch,
                model,
                optimizer,
                dataset["ood_val_loader"],
                ood_val_loss_computer,
                logger,
                None,
                args,
                is_training=False,
            )
            if compute_tracker is not None:
                compute_tracker.end_phase(epoch, "ood_val")

            # Use id_val for scheduler/adjustment (replaces val_loss_computer)
            val_loss_computer = id_val_loss_computer

            id_val_acc = id_val_loss_computer.avg_acc.item()
            ood_val_acc = ood_val_loss_computer.avg_acc.item()

            # Epoch summary
            logger.write(
                f"Epoch {epoch} Summary: "
                f"train_loss={train_epoch_stats['avg_loss']:.4f}, "
                f"train_acc={train_epoch_stats['avg_acc']:.4f}, "
                f"id_val_acc={id_val_acc:.4f}, "
                f"ood_val_acc={ood_val_acc:.4f}\n"
            )

            # Write to metrics.csv
            metrics_file.write(
                f"{epoch},{train_epoch_stats['avg_loss']:.6f},"
                f"{train_epoch_stats['avg_acc']:.4f},"
                f"{id_val_acc:.4f},{ood_val_acc:.4f}\n"
            )
            metrics_file.flush()

            # Save best models
            if id_val_acc > best_id_val_acc:
                best_id_val_acc = id_val_acc
                torch.save(
                    model.state_dict(),
                    os.path.join(args.log_dir, "best_id_model.pth"),
                )
                logger.write(f"Best ID model saved at epoch {epoch}\n")

            if ood_val_acc > best_ood_val_acc:
                best_ood_val_acc = ood_val_acc
                torch.save(
                    model.state_dict(),
                    os.path.join(args.log_dir, "best_ood_model.pth"),
                )
                logger.write(f"Best OOD model saved at epoch {epoch}\n")
        else:
            logger.write("\nValidation:\n")
            val_loss_computer = LossComputer(
                criterion,
                is_robust=args.robust,
                dataset=dataset["val_data"],
                step_size=args.robust_step_size,
                alpha=args.alpha,
            )
            if compute_tracker is not None:
                compute_tracker.start_phase(
                    epoch,
                    "val",
                    num_samples=len(dataset["val_loader"].dataset),
                    num_batches=len(dataset["val_loader"]),
                )
            run_epoch(
                epoch,
                model,
                optimizer,
                dataset["val_loader"],
                val_loss_computer,
                logger,
                None,
                args,
                is_training=False,
            )
            if compute_tracker is not None:
                compute_tracker.end_phase(epoch, "val")

            curr_val_acc = val_loss_computer.avg_acc.item()

            # Epoch summary
            logger.write(
                f"Epoch {epoch} Summary: "
                f"train_loss={train_epoch_stats['avg_loss']:.4f}, "
                f"train_acc={train_epoch_stats['avg_acc']:.4f}, "
                f"val_acc={curr_val_acc:.4f}\n"
            )

            # Write to metrics.csv
            metrics_file.write(
                f"{epoch},{train_epoch_stats['avg_loss']:.6f},"
                f"{train_epoch_stats['avg_acc']:.4f},"
                f"{curr_val_acc:.4f}\n"
            )
            metrics_file.flush()

            # Save best model
            if curr_val_acc > best_val_acc:
                best_val_acc = curr_val_acc
                torch.save(
                    model.state_dict(),
                    os.path.join(args.log_dir, "best_model.pth"),
                )
                logger.write(f"Best model saved at epoch {epoch}\n")

        # Inspect learning rates
        for param_group in optimizer.param_groups:
            curr_lr = param_group["lr"]
            logger.write("Current lr: %f\n" % curr_lr)

        if args.scheduler and args.model != "bert":
            if args.robust:
                val_loss, _ = val_loss_computer.compute_robust_loss_greedy(
                    val_loss_computer.avg_group_loss, val_loss_computer.avg_group_loss
                )
            else:
                val_loss = val_loss_computer.avg_actual_loss
            scheduler.step(val_loss)

        if args.automatic_adjustment:
            gen_gap = (
                val_loss_computer.avg_group_loss - train_loss_computer.exp_avg_loss
            )
            adjustments = gen_gap * torch.sqrt(train_loss_computer.group_counts)
            train_loss_computer.adj = adjustments
            logger.write("Adjustments updated\n")
            for group_idx in range(train_loss_computer.n_groups):
                logger.write(
                    f"  {train_loss_computer.get_group_name(group_idx)}:\t"
                    f"adj = {train_loss_computer.adj[group_idx]:.3f}\n"
                )
        logger.write("\n")

    metrics_file.close()

    if compute_tracker is not None:
        compute_tracker.end_training()

    # Final test evaluation with best models, per-sample JSON, and test_results.csv
    if dataset["test_data"] is not None:
        if use_4way:
            models_to_eval = [
                ("ID", "best_id_model.pth", "best_id_val"),
                ("OOD", "best_ood_model.pth", "best_ood_val"),
            ]
        else:
            models_to_eval = [("val", "best_model.pth", "best_val")]

        test_results = []

        for label, model_path, result_key in models_to_eval:
            path = os.path.join(args.log_dir, model_path)
            if not os.path.exists(path):
                logger.write(f"\nNo {label} best model found, skipping final test.\n")
                continue
            logger.write(f"\n=== Final Test Evaluation (Best {label} Val Model) ===\n")
            model.load_state_dict(torch.load(path, map_location=get_device()))
            final_test_lc = LossComputer(
                criterion,
                is_robust=args.robust,
                dataset=dataset["test_data"],
                step_size=args.robust_step_size,
                alpha=args.alpha,
            )
            test_stats = run_epoch(
                0,
                model,
                None,
                dataset["test_loader"],
                final_test_lc,
                logger,
                None,
                args,
                is_training=False,
            )

            test_results.append((result_key, test_stats["avg_acc"]))

            # Save per-sample predictions as JSON
            json_name = f"predictions_best_{label.lower()}_model.json"
            json_path = os.path.join(args.log_dir, json_name)
            log_predictions_json(model, dataset["test_data"], json_path, args)
            logger.write(f"Predictions saved to {json_path}\n")

        # Write test_results.csv
        if test_results:
            results_path = os.path.join(args.log_dir, "test_results.csv")
            with open(results_path, "w") as f:
                f.write("model_selection,test_acc\n")
                for key, acc in test_results:
                    f.write(f"{key},{acc:.4f}\n")
            logger.write(f"\nTest results saved to {results_path}\n")
