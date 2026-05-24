# YOLO11-M Baseline Training Record

This file records the baseline configuration used for WaterScenes YOLO11-M
training.

## Experiment

```text
Experiment: YOLO11-M baseline
Run name: yolo11m_baseline
Project: runs/aefc_yolo11
Backbone/pretrained weight: weights/yolo11m.pt
Custom modules: none
```

## Dataset

```text
Dataset config: configs/waterscenes_full.yaml
Train split: train.txt -> images/train
Val split:   val.txt   -> images/val
Test split:  test.txt  -> images/test
```

Classes:

| id | class |
|---:|---|
| 0 | pier |
| 1 | buoy |
| 2 | sailor |
| 3 | ship |
| 4 | boat |
| 5 | vessel |
| 6 | kayak |

## Training Hyperparameters

| Parameter | Value |
|---|---:|
| imgsz | 1920 |
| epochs | 200 |
| global batch | 32 |
| GPUs | 4 |
| per-GPU batch | 8 |
| optimizer | AdamW |
| lr0 | 0.001 |
| lrf | 0.01 |
| final lr | 0.00001 |
| weight_decay | 0.0005 |
| momentum | 0.937 |
| warmup_epochs | 3 |
| warmup_momentum | 0.8 |
| warmup_bias_lr | 0.1 |
| cos_lr | true |
| amp | false |
| seed | 42 |
| log interval | 100 batches |

## Training Command

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

## Epoch Step Count

```text
train images = 37884
global batch = 32
steps per epoch = ceil(37884 / 32) = 1184
```

The progress bar value `0/1184 ... 1184/1184` is the global DDP step count.

## Recorded Validation Result

```text
Images: 10824
Instances: 40517
Precision: 0.711
Recall: 0.562
mAP50: 0.629
mAP50-95: 0.373
```
