# Baseline Experiment Plan

This repository only covers the YOLO11-M baseline experiment.

## Required Run

| Run | Model | Weight | Dataset | Purpose |
|---|---|---|---|---|
| yolo11m_baseline | YOLO11-M | weights/yolo11m.pt | WaterScenes fixed split | Baseline detector result |

## Training

```bash
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

## Evaluation

Regular test split:

```bash
yolo detect val \
  model=runs/aefc_yolo11/yolo11m_baseline/weights/best.pt \
  data=configs/waterscenes_full.yaml \
  imgsz=1920 \
  batch=32 \
  device=0 \
  split=test \
  project=runs/aefc_yolo11_eval \
  name=yolo11m_baseline_test
```

Adverse lighting:

```bash
yolo detect val \
  model=runs/aefc_yolo11/yolo11m_baseline/weights/best.pt \
  data=configs/waterscenes_adverse_lighting.yaml \
  imgsz=1920 \
  batch=32 \
  device=0 \
  split=val \
  project=runs/aefc_yolo11_eval \
  name=yolo11m_baseline_adverse_lighting
```

Adverse weather:

```bash
yolo detect val \
  model=runs/aefc_yolo11/yolo11m_baseline/weights/best.pt \
  data=configs/waterscenes_adverse_weather.yaml \
  imgsz=1920 \
  batch=32 \
  device=0 \
  split=val \
  project=runs/aefc_yolo11_eval \
  name=yolo11m_baseline_adverse_weather
```

## Metrics

Record at least:

```text
Precision
Recall
mAP50
mAP50-95
```

Recorded validation result:

```text
Precision = 0.711
Recall = 0.562
mAP50 = 0.629
mAP50-95 = 0.373
```
