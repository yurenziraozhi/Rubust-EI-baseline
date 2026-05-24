# YOLO11-M WaterScenes Baseline

This repository is a reconstructed baseline-only training package for the
WaterScenes YOLO11-M experiment. It intentionally does not include UIAE, EAFC,
MDCT, or any full AEFC experimental modules.

The baseline was originally trained before the project was placed under Git, so
this repository is rebuilt from the saved training configuration, logs, and
workspace files.

## Server Layout

Keep the dataset folders next to `AEFC-YOLO11/`:

```text
workdir/
|-- AEFC-YOLO11/
|-- image/
|-- detection/
|   `-- yolo/
|-- train.txt
|-- val.txt
|-- test.txt
|-- adverse_lighting.txt
`-- adverse_weather.txt
```

Raw data and generated datasets are ignored by Git. This repository only keeps
code, configs, the YOLO11-M pretrained weight, and one sample image.

## Prepare Dataset

Run from `AEFC-YOLO11/`:

```bash
pip install -r requirements.txt

python tools/prepare_waterscenes_yolo.py \
  --root .. \
  --image-dir image \
  --label-dir detection/yolo \
  --train-list train.txt \
  --val-list val.txt \
  --test-list test.txt \
  --lighting-list adverse_lighting.txt \
  --weather-list adverse_weather.txt \
  --out datasets/waterscenes_yolo \
  --mode hardlink
```

The fixed split sizes are:

```text
train: 37884
val:   10824
test:   5412
```

## Train Baseline

The baseline uses the official Ultralytics YOLO11-M detector initialized from
`weights/yolo11m.pt`.

```bash
mkdir -p logs

nohup python tools/train_baseline.py \
  --cfg configs/train_baseline_1920.yaml \
  --device 0,1,2,3 \
  --project runs/aefc_yolo11 \
  --name yolo11m_baseline \
  --log-dir logs \
  --log-file logs/yolo11m_baseline.log \
  --log-interval 100 \
  --save-period -1 \
  --plots false \
  > logs/yolo11m_baseline.nohup.out 2>&1 &
```

Equivalent helper:

```bash
bash tools/train_ddp_nohup.sh
```

## Baseline Config

```yaml
model: weights/yolo11m.pt
data: configs/waterscenes_full.yaml
imgsz: 1920
epochs: 200
batch: 32
optimizer: AdamW
lr0: 0.001
lrf: 0.01
weight_decay: 0.0005
momentum: 0.937
warmup_epochs: 3
warmup_momentum: 0.8
warmup_bias_lr: 0.1
cos_lr: true
workers: 8
device: 0,1,2,3
amp: false
seed: 42
```

With 37,884 train images and global batch size 32, each epoch has:

```text
ceil(37884 / 32) = 1184 steps
```

## Outputs

The training script writes:

```text
logs/yolo11m_baseline.log
logs/yolo11m_baseline_epoch_metrics.csv
logs/yolo11m_baseline.nohup.out
```

Ultralytics writes model checkpoints under:

```text
runs/aefc_yolo11/yolo11m_baseline/weights/
```

Only `best.pt` and `last.pt` are kept by default when `save_period=-1`.

## Recorded Baseline Result

From the saved validation output:

```text
Images: 10824
Instances: 40517
Precision: 0.711
Recall: 0.562
mAP50: 0.629
mAP50-95: 0.373
```

Use this result as the comparison point for later UIAE, EAFC, and full AEFC
experiments.
