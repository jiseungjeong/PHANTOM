# Data Collection Protocol

How to record a new subject so their data is consistent with the existing cohort and
is picked up automatically by the analysis pipeline.

## 1. Environment

- **Wear shoes.** The evaluation cohort was recorded with shoes on; barefoot or slipper
  recordings change the IMU signature and are excluded from the cohort.
- Use the same kind of place and floor as the existing recordings where possible.

## 2. Devices

All three IMUs must be connected and streaming before you start.

| Device | Placement |
|---|---|
| AirPods Pro | Head (both buds in) |
| Apple Watch | Lead wrist (usually the left) |
| iPhone | Held in the rear hand |

AirPods only report head motion while they are the active audio route. The app handles
this, but confirm in the Raw Stream tab that all three sources are receiving data.

## 3. Actions

Record each of the 10 single actions **3 times** (30 recordings per subject). Each
recording is **one sync jump, then the same action repeated 5 times**.

| # | Label (exactly as shown in the app) |
|---|---|
| 1 | Lead Jab |
| 2 | Lead Hook |
| 3 | Lead Uppercut |
| 4 | Rear Jab |
| 5 | Rear Hook |
| 6 | Rear Uppercut |
| 7 | Left Slip |
| 8 | Right Slip |
| 9 | Left Roll |
| 10 | Right Roll |

Tips for clean segmentation (the peak detector assumes evenly spaced, similar-sized
repetitions):

- Make the sync jump large and distinct from the actions.
- Keep a steady rhythm and intensity across the 5 repetitions.
- Pause briefly between repetitions so the recovery motion is not picked up as a peak.
- Perform slips and rolls clearly; weak defensive moves may not produce a peak.

## 4. File naming

The pipeline parses subject and label from the file name:

```
{SUBJECT}_{Label}_{YYYYMMDD}_{HHMMSS}.csv          # full-sensor (all 3 devices)
{SUBJECT}_{Label}_Watch_{YYYYMMDD}_{HHMMSS}.csv    # Watch-derived copy
e.g. KIM_Lead Jab_20260610_143022.csv
```

- `{SUBJECT}`: an upper-case name that does not collide with an existing subject.
- `{Label}`: exactly one of the labels above, including the space. A typo creates a new class.

The app saves files in this format already.

## 5. Location

```
Data/{subject_lowercase}_data/
e.g. Data/kim_data/KIM_Lead Jab_20260610_143022.csv
```

A new folder joins the LOSO cohort automatically (`evaluation.split_cohort`); no code
changes are needed. `Data/` is git-ignored.

## 6. Re-running the pipeline

From the repository root:

```bash
# 1) Inspect the data (AirPods validity, label consistency, peak counts)
python3 analysis/five_subject_active_window_classifier/phase0_inspect.py

# 2) Rebuild the segmented dataset (single_dataset.json)
python3 analysis/five_subject_active_window_classifier/phase1_segment.py

# 3) Re-run the LOSO evaluation (new subject included automatically)
python3 analysis/five_subject_active_window_classifier/evaluation.py

# 4) (optional) Re-run the model comparison
python3 analysis/five_subject_active_window_classifier/phase3_models.py
```

Check the Phase 0 output for missing AirPods data, label typos and unexpected peak counts
before trusting the evaluation numbers.

## 7. Checklist

- [ ] Shoes on
- [ ] AirPods, Watch and iPhone connected and streaming
- [ ] 10 actions x 3 recordings = 30 files (each: jump, then 5 repetitions)
- [ ] File and folder names follow `Data/{name}_data/{NAME}_{Label}_...`
- [ ] Phase 0 reports no exclusions, consistent labels and 5 peaks per file
