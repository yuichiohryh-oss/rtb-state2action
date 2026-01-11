# rtb-state2action

Hand recognition pipeline for a local scrcpy window capture or recorded mp4 input. The repository collects hand crops, labels them into 8 classes, trains a lightweight CNN, and streams inference results as an `in_hand` vector suitable for downstream state-to-action models.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Notes:
- The last line installs the official CUDA 12.8 (cu128) wheels. Skip it if you want CPU-only PyTorch.
- `dxcam` is optional and only needed for higher-performance real-time capture. If it is missing, the capture code falls back to `mss`.
  - Install it when needed: `pip install dxcam`

## Quickstart（最短手順）

```powershell
python -m scripts.collect_hand_crops --window-title scrcpy_game --interval-ms 250 --hand-y-ratio 0.72 --hand-height-ratio 0.26
python -m scripts.label_hand_crops --crops-jsonl data/hand_crops.jsonl
python -m scripts.train_hand_cnn --labels data/hand_labels.jsonl
python -m scripts.infer_hand --window-title scrcpy_game --model runs/<run_id>/model.pt
```

mp4????:

```powershell
python -m scripts.collect_hand_crops --video samples/scrcpy_001.mp4 --interval-ms 250
python -m scripts.collect_hand_crops --video samples/scrcpy_001.mp4 --preview
python -m scripts.infer_hand --video samples/scrcpy_001.mp4 --model runs/<run_id>/model.pt --max-frames 200
```

Snipping Tool ??? scrcpy ??????????????? ROI ????????????
`samples/` ? mp4 ?????? Git ???????????`.gitignore` ????

ROIの調整:
- `y_ratio`: ウィンドウ高さに対するROI開始位置の比率（例: 0.72）
- `height_ratio`: ウィンドウ高さに対するROIの高さ比率（例: 0.26）

プレビュー確認（保存なし、`q`で終了）:

```powershell
python -m scripts.collect_hand_crops --window-title scrcpy_game --preview
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
- For `--video`, `--window-title` is ignored and `--interval-ms` does not affect sampling.

## CLI Details

`collect_hand_crops` supports ROI parameters:

```powershell
python -m scripts.collect_hand_crops --window-title scrcpy_game --interval-ms 250 --hand-y-ratio 0.72 --hand-height-ratio 0.26
```

Video sampling options:

```powershell
python -m scripts.collect_hand_crops --video samples/scrcpy_001.mp4 --video-fps 4
```

`--video-fps` samples frames at the target rate by skipping frames based on their timestamps (`CAP_PROP_POS_MSEC`). If timestamps are unavailable, the sampler falls back to the source fps.

`infer_hand` can append to a JSONL state stream:

```powershell
python -m scripts.infer_hand --window-title scrcpy_game --model runs/<run_id>/model.pt --state-out data/state_stream.jsonl
```

## Testing

```powershell
pytest
```
