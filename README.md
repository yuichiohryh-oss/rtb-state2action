# rtb-state2action

Hand recognition pipeline for a local scrcpy window capture. The repository collects hand crops, labels them into 8 classes, trains a lightweight CNN, and streams inference results as an `in_hand` vector suitable for downstream state-to-action models.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Quick Start

```powershell
python -m scripts.collect_hand_crops --window-title scrcpy_game --interval-ms 250
python -m scripts.label_hand_crops --crops-jsonl data/hand_crops.jsonl
python -m scripts.train_hand_cnn --labels data/hand_labels.jsonl
python -m scripts.infer_hand --window-title scrcpy_game --model runs/<run_id>/model.pt
```

## Data Layout

```
data/
  hand_crops/
    <session_id>/
      t_<ms>_slot0.png
  hand_crops.jsonl
  hand_labels.jsonl
runs/
  <run_id>/
    model.pt
    metrics.json
```

## Notes

- The window title must match the scrcpy window title.
- `HAND_ROI` is defined as the lower portion of the window and split into 4 slots.
- The ROI includes extra vertical margin to tolerate selected-card lift.
- All outputs are local-only; image data is not tracked in Git.

## CLI Details

`collect_hand_crops` supports ROI parameters:

```powershell
python -m scripts.collect_hand_crops --window-title scrcpy_game --interval-ms 250 --y-ratio 0.72 --height-ratio 0.26
```

`infer_hand` can append to a JSONL state stream:

```powershell
python -m scripts.infer_hand --window-title scrcpy_game --model runs/<run_id>/model.pt --state-out data/state_stream.jsonl
```

## Testing

```powershell
pytest
```
