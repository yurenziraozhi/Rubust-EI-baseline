import argparse
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path

import yaml
from ultralytics import YOLO


def str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y", "on"}


def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def pick(args, cfg, key, default=None):
    value = getattr(args, key, None)
    return cfg.get(key, default) if value is None else value


def rank0():
    return int(os.environ.get("RANK", "0")) == 0


def now():
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


class JsonlLogger:
    def __init__(self, log_file, csv_file, interval):
        self.log_file = Path(log_file)
        self.csv_file = Path(csv_file)
        self.interval = int(interval)
        self.start = time.time()
        if rank0():
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            self.csv_file.parent.mkdir(parents=True, exist_ok=True)
            self.log_file.write_text("", encoding="utf-8")
            with self.csv_file.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["epoch", "precision", "recall", "map50", "map50_95", "box_loss", "cls_loss", "dfl_loss"],
                )
                writer.writeheader()

    def write(self, payload):
        if not rank0():
            return
        payload = {"time": now(), **payload}
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def append_epoch_csv(self, row):
        if not rank0():
            return
        with self.csv_file.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["epoch", "precision", "recall", "map50", "map50_95", "box_loss", "cls_loss", "dfl_loss"],
            )
            writer.writerow(row)


def tensor_to_float(value):
    try:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "item"):
            return float(value.item())
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/train_baseline_1920.yaml")
    parser.add_argument("--model")
    parser.add_argument("--data")
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--device")
    parser.add_argument("--project")
    parser.add_argument("--name")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--log-file")
    parser.add_argument("--log-interval", type=int)
    parser.add_argument("--save-period", type=int)
    parser.add_argument("--plots")
    args = parser.parse_args()

    cfg = load_cfg(args.cfg)
    name = pick(args, cfg, "name", "yolo11m_baseline")
    log_dir = Path(args.log_dir)
    log_file = args.log_file or str(log_dir / f"{name}.log")
    csv_file = str(log_dir / f"{name}_epoch_metrics.csv")
    logger = JsonlLogger(log_file, csv_file, pick(args, cfg, "log_interval", 100))

    train_args = {
        "data": pick(args, cfg, "data", "configs/waterscenes_full.yaml"),
        "imgsz": int(pick(args, cfg, "imgsz", 1920)),
        "epochs": int(pick(args, cfg, "epochs", 200)),
        "batch": int(pick(args, cfg, "batch", 32)),
        "device": pick(args, cfg, "device", "0,1,2,3"),
        "project": pick(args, cfg, "project", "runs/aefc_yolo11"),
        "name": name,
        "optimizer": cfg.get("optimizer", "AdamW"),
        "lr0": float(cfg.get("lr0", 0.001)),
        "lrf": float(cfg.get("lrf", 0.01)),
        "weight_decay": float(cfg.get("weight_decay", 0.0005)),
        "momentum": float(cfg.get("momentum", 0.937)),
        "warmup_epochs": int(cfg.get("warmup_epochs", 3)),
        "warmup_momentum": float(cfg.get("warmup_momentum", 0.8)),
        "warmup_bias_lr": float(cfg.get("warmup_bias_lr", 0.1)),
        "cos_lr": str2bool(cfg.get("cos_lr", True)),
        "workers": int(cfg.get("workers", 8)),
        "amp": str2bool(cfg.get("amp", False)),
        "seed": int(cfg.get("seed", 42)),
        "save_period": int(pick(args, cfg, "save_period", -1)),
        "plots": str2bool(pick(args, cfg, "plots", False)),
        "exist_ok": bool(cfg.get("exist_ok", False)),
        "verbose": bool(cfg.get("verbose", False)),
    }
    model_path = pick(args, cfg, "model", "weights/yolo11m.pt")

    logger.write(
        {
            "event": "run_start",
            "model": model_path,
            "pretrained_backbone": model_path,
            **train_args,
            "enabled_modules": {"uiae": False, "eafc": False, "mdct": False},
        }
    )

    model = YOLO(model_path)

    def on_train_batch_end(trainer):
        batch_i = int(getattr(trainer, "batch_i", -1)) + 1
        total = len(getattr(trainer, "train_loader", []) or [])
        if batch_i <= 0 or (batch_i % logger.interval != 0 and batch_i != total):
            return
        loss_items = getattr(trainer, "loss_items", None)
        losses = {}
        if loss_items is not None:
            values = loss_items.tolist() if hasattr(loss_items, "tolist") else list(loss_items)
            for key, value in zip(["box_loss", "cls_loss", "dfl_loss"], values):
                losses[key] = tensor_to_float(value)
        logger.write(
            {
                "event": "train_batch",
                "epoch": int(getattr(trainer, "epoch", 0)) + 1,
                "batch": batch_i,
                "total_batch": total,
                **losses,
            }
        )

    def on_fit_epoch_end(trainer):
        metrics = getattr(trainer, "metrics", {}) or {}
        loss_items = getattr(trainer, "tloss", None)
        losses = {}
        if loss_items is not None:
            values = loss_items.tolist() if hasattr(loss_items, "tolist") else list(loss_items)
            for key, value in zip(["box_loss", "cls_loss", "dfl_loss"], values):
                losses[key] = tensor_to_float(value)
        row = {
            "epoch": int(getattr(trainer, "epoch", 0)) + 1,
            "precision": tensor_to_float(metrics.get("metrics/precision(B)", metrics.get("precision"))),
            "recall": tensor_to_float(metrics.get("metrics/recall(B)", metrics.get("recall"))),
            "map50": tensor_to_float(metrics.get("metrics/mAP50(B)", metrics.get("mAP50"))),
            "map50_95": tensor_to_float(metrics.get("metrics/mAP50-95(B)", metrics.get("mAP50-95"))),
            **losses,
        }
        logger.write({"event": "epoch_end", **row})
        logger.append_epoch_csv(row)

    def on_train_end(trainer):
        logger.write({"event": "run_end", "status": "finished", "elapsed_sec": round(time.time() - logger.start, 2)})

    model.add_callback("on_train_batch_end", on_train_batch_end)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    model.add_callback("on_train_end", on_train_end)

    try:
        model.train(**train_args)
    except Exception as exc:
        logger.write({"event": "run_end", "status": "failed", "elapsed_sec": round(time.time() - logger.start, 2), "error": repr(exc)})
        raise


if __name__ == "__main__":
    main()
