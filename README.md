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
- `x_margin_ratio`: ウィンドウ幅に対して左右からカットする比率
- `x_offset_ratio`: ウィンドウ幅に対する比率でROIを左右にシフト（+は右、-は左）
- slot3が切れるときは x_offset を正方向に調整する

推奨例:

```powershell
python -m scripts.collect_hand_crops --window-title scrcpy_game --interval-ms 250 --hand-y-ratio 0.75 --hand-height-ratio 0.33 --x-margin-ratio 0.18 --x-offset-ratio 0.04
```

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

PowerShell redirect examples (stdout -> file):

```powershell
python -m scripts.infer_hand --window-title scrcpy_game --model runs/<run_id>/model.pt | Out-File -FilePath data/infer_hand.txt
```

`Out-File` produces UTF-16LE by default; the extractor handles UTF-8/UTF-16LE inputs.

Regenerate 231410 with slot enforcement and slot ids:

```powershell
cd repoRoot
.\.venv\Scripts\Activate.ps1
python -m scripts.infer_hand --video ... --video-fps 10 --model ... --enforce-unique-cards --emit-slots > out.hand.txt
```

Regenerate 231410 via `tools/regen_one.ps1` with extra infer flags:

```powershell
cd repoRoot
.\.venv\Scripts\Activate.ps1
.\tools\regen_one.ps1 -Video .\samples\batch3\hog_yt_2026-01-11_231410.mp4 -Fps 10 -OutDir data\batch3_fps10_enforced -InferArgs "--enforce-unique-cards","--emit-slots"
```

`extract_actions_from_hand` converts the JSONL stream into action events:

```powershell
python -m scripts.extract_actions_from_hand --in data/infer_hand.txt --out data/actions.jsonl
python -m scripts.extract_actions_from_hand --in - --out data/actions.jsonl
```

Output schema (`actions.jsonl`):

```json
{"t_ms": 1000, "card_id": 1, "event": "play", "confidence": 0.8, "hand_before": [1, 2, 3, 4], "hand_after": [2, 3, 4, 5], "notes": "confirm=2 pre_hold=2/2 invalid_window=0"}
```

Label play positions from board diffs (`actions.jsonl` -> `actions_with_pos.jsonl`):

```powershell
cd repoRoot
.\.venv\Scripts\Activate.ps1
python .\tools\label_position_from_diff.py --video .\samples\batch3\hog_yt_2026-01-11_231410.mp4 --actions .\data\batch3_fps10_enforced\actions.jsonl --out .\data\batch3_fps10_enforced\actions_with_pos.jsonl --debug-dir .\data\batch3_fps10_enforced\pos_debug --dt-ms 300 --grid-w 18 --grid-h 11 --thr 18 --after-stability 3 --self-side-only --self-side-ratio 0.52 --component-score sum
```

If `--roi` is omitted, an interactive ROI picker opens (use `--roi-pick-tms` to choose the frame time).
If enemy-side overlays are being picked, keep `--self-side-only` enabled and adjust `--self-side-ratio` (defaults to 0.52).

## Tap teacher pipeline

Prereqs:
- `adb` and `scrcpy` available in PATH
- USB debugging enabled on the device

One command capture + label:

```powershell
powershell -ExecutionPolicy Bypass -File tools/run_tap_teacher.ps1 -Seconds 180 -OutDir runs
```

If `adb` / `scrcpy` are not on PATH, pass `-Adb` / `-Scrcpy` with full paths:

```powershell
.\tools\run_tap_teacher.ps1 `
  -Seconds 180 `
  -OutDir runs `
  -Scrcpy "C:\Users\...\scrcpy.exe"
```

Outputs (created under `runs/<run_id>/`):

```
video.mp4
taps.csv
meta.json
actions_tap.jsonl
actions_tap_pos.jsonl
debug_pos/
```

## Position model (cell classification)

Use the tap teacher output (`actions_tap_pos.jsonl`) to train a minimal cell classifier.

Build a manifest from a run:

```powershell
python tools\build_pos_dataset.py ^
  --input runs\smoke\actions_tap_pos.jsonl ^
  --out data\pos\smoke_manifest.jsonl ^
  --min-conf 0.7
```

Train the model:

```powershell
python train_pos_model.py ^
  --train-manifest data\pos\smoke_manifest.jsonl ^
  --val-manifest data\pos\smoke_manifest.jsonl ^
  --grid-w 18 --grid-h 11 ^
  --img-size 224 ^
  --batch-size 32 ^
  --epochs 5 ^
  --lr 1e-3 ^
  --out runs\pos_train\exp001
```

PowerShell runner (builds manifest then trains):

```powershell
.\tools\run_train_pos.ps1 -InputJsonl runs\smoke\actions_tap_pos.jsonl -OutDir runs\pos_train\smoke -Epochs 5
```

If the manifest has 0 samples:
- Rows with `pos.cell_id` set to null are skipped.
- Provide `--debug-dir` (or `-DebugDir`) when `paths.diff` is missing so diff images can be resolved.

Outputs:

```
config.json
metrics.jsonl
model.pt
```

## Mouse-based tap teacher capture (Windows)

Prereqs:
- `scrcpy` available on disk (pass with `-Scrcpy` if not on PATH).
- Python deps: `pywin32`, `pynput`, and `opencv-python` (recommended for frame metadata).

Run the end-to-end pipeline (record + clicks + diff labeling):

```powershell
.\tools\run_tap_teacher_mouse.ps1 -Seconds 10 -OutDir runs -Scrcpy "C:\Users\yuichi\Documents\tool\scrcpy-win64-v3.3.4\scrcpy.exe"
```

During capture, click inside the scrcpy window. Clicks are recorded in video frame coordinates.
Playback is required for mouse capture; the window is shown by default. Use `--no-playback` (or `-NoPlayback` in the PowerShell runner) only when you want a hidden recording without click labels.
Audio recording is always disabled (`--no-audio`).

Outputs (created under `runs/run_YYYYmmdd_HHMMSS/`):

```
video.mp4
taps.csv
meta.json
actions_tap.jsonl
actions_tap_pos.jsonl
debug_pos/
```

`build_state_role_dataset` joins hand frames and action events into state-role pairs:

```powershell
python -m scripts.build_state_role_dataset --hand data/result_full.txt --actions data/actions_full.jsonl --out data/state_role.jsonl --state-offset-ms 1000 --max-gap-ms 1500
```

Attach a video stem (mp4 filename without extension) so later splits can group by video:

```powershell
python -m scripts.build_state_role_dataset --hand data/batch1/result.txt --actions data/batch1/actions.jsonl --out data/state_role_batch1.jsonl --stem batch1
python -m scripts.build_state_role_dataset --hand data/batch2/result.txt --actions data/batch2/actions.jsonl --out data/state_role_batch2.jsonl --stem batch2
python -m scripts.build_state_role_dataset --hand data/batch3/result.txt --actions data/batch3/actions.jsonl --out data/state_role_batch3.jsonl --stem batch3
python -m scripts.build_state_role_dataset --hand data/batch4/result.txt --actions data/batch4/actions.jsonl --out data/state_role_batch4.jsonl --stem batch4
```

Include history in the state:

```powershell
python -m scripts.build_state_role_dataset --hand data/result_full.txt --actions data/actions_full.jsonl --out data/state_role.jsonl --include-prev-action
```

History options (`--history`):
- 0: no history (v1)
- 1: prev1 only (v2, default)
- 2: prev1 + prev2 (v3)
Note: `--include-prev-action` is kept for backward compatibility; `--history` >= 1 includes `prev_action` fields.

v3 dataset example (prev1 + prev2):

```powershell
python -m scripts.build_state_role_dataset --hand data/result_full.txt --actions data/actions_full.jsonl --out data/state_role.jsonl --include-prev-action --history 2
```

Output schema (`state_role.jsonl`):

```json
{"t_ms_event": 20266, "t_ms_state": 19266, "in_hand_state": [0, 1, 1, 1, 0, 0, 0, 1], "role": 1, "state_source": "nearest_frame", "prev_action": 4, "prev_action_onehot": [0, 0, 0, 1, 0, 0, 0, 0]}
```

Notes:
- `actions.t_ms` is closer to the confirmed play time, so `state_offset_ms` shifts the state earlier.
- Use `--include-debug` to include `confidence`, `hand_before`, and `hand_after` in the output.

## Proposal Model (state -> role)

The minimal proposal model predicts the next card role (1..8) from the hand state (v1), hand+prev_action (v2), or hand+prev_action+prev2_action (v3).

Train:

```powershell
python -m scripts.train_proposal_model --data data/state_role.jsonl --epochs 30 --batch-size 64 --lr 1e-3 --seed 42 --val-split 0.2
```

Train with group split (same stem stays in the same split):

```powershell
python -m scripts.train_proposal_model --data data/state_role.jsonl --epochs 30 --batch-size 64 --lr 1e-3 --seed 42 --val-split 0.2 --split-mode group
```

Enable balanced class weights (based on the train split role counts):

```powershell
python -m scripts.train_proposal_model --data data/state_role.jsonl --epochs 30 --batch-size 64 --lr 1e-3 --seed 42 --val-split 0.2 --class-weight balanced
```

Artifacts are saved to `runs/YYYYMMDD_HHMMSS_proposal/`:

```
model.pt
best_model.pt
config.json
metrics.json
```

Infer (single hand):

```powershell
python -m scripts.infer_proposal_model --model runs/<run_id>/model.pt --hand "0,1,1,1,1,0,0,0" --topk 3
```

Infer (hand + prev_action):

```powershell
python -m scripts.infer_proposal_model --model runs/<run_id>/model.pt --hand "0,1,1,1,1,0,0,0" --prev 4 --topk 3
```

Infer (hand + prev_action + prev2_action):

```powershell
python -m scripts.infer_proposal_model --model runs/<run_id>/model.pt --hand "0,1,1,1,1,0,0,0" --prev 4 --prev2 2 --topk 3
```

Infer from `state_role.jsonl` (first N samples):

```powershell
python -m scripts.infer_proposal_model --model runs/<run_id>/model.pt --from-state-role data/state_role.jsonl --n 20 --topk 3
```

Evaluate (val split by default, same split seed as training). Use `best_model.pt` to reproduce peak validation metrics:

```powershell
python -m scripts.eval_proposal_model --model runs/<run_id>/best_model.pt --data data/state_role.jsonl --split val --val-split 0.2 --seed 42 --topk 3 --out runs/<run_id>_proposal_eval/metrics.json
```

Evaluate with group split (same split mode as training):

```powershell
python -m scripts.eval_proposal_model --model runs/<run_id>/best_model.pt --data data/state_role.jsonl --split val --val-split 0.2 --seed 42 --split-mode group --topk 3
```

Evaluate a fixed holdout stem (e.g., batch4):

```powershell
python -m scripts.eval_proposal_model --model runs/<run_id>/best_model.pt --data data/state_role.jsonl --holdout-stems batch4 --topk 3
```

Evaluate on all samples (no split) or compare baselines:

```powershell
python -m scripts.eval_proposal_model --model runs/<run_id>/model.pt --data data/state_role.jsonl --split all --topk 3
python -m scripts.eval_proposal_model --data data/state_role.jsonl --baseline mostfreq --split val --val-split 0.2 --seed 42
python -m scripts.eval_proposal_model --data data/state_role.jsonl --baseline random --split val --val-split 0.2 --seed 42
```

Input schema (`state_role.jsonl`):

```json
{"in_hand_state": [0, 1, 1, 1, 1, 0, 0, 0], "role": 1, "prev_action": 4, "prev_action_onehot": [0, 0, 0, 1, 0, 0, 0, 0]}
```

Notes:
- `role` is 1..8 (external I/O keeps 1..8, internal model uses 0..7).
- v1 uses hand-only state; v2 appends `prev_action_onehot` to reach 16 dims; v3 appends `prev2_action_onehot` to reach 24 dims.
- `--class-weight balanced` works with v1/v2/v3 datasets and clips weights at 5.0 to avoid extreme values.
- Tip: if predictions collapse to a single class, try `--class-weight balanced`.
- Recommended workflow: assign `--stem` per batch, train with `--split-mode group` on batches 1-3, then evaluate with `--holdout-stems batch4` to measure generalization.

`label_hand_crops` shows key bindings and card mappings in the top-left corner; use `--no-help` to hide them. The fixed deck is the 2.6 hog list:
- 1: HOG_RIDER
- 2: MUSKETEER
- 3: CANNON
- 4: ICE_GOLEM
- 5: SKELETONS
- 6: ICE_SPIRIT
- 7: FIREBALL
- 8: THE_LOG

## Testing

```powershell
pytest
```
