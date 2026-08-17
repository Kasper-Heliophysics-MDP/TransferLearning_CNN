# Solar Radio Burst Detection (SunRISE Ground Radio Lab)

Train a solar radio burst (Type II / III / V) **detector** on e-Callisto network
spectrograms, then transfer it to the local receiver. COCO-pretrained YOLOv8 with
object detection over fixed 15-minute windows.

**For results and conclusions, see [`RESULTS.md`](RESULTS.md).** This file covers
how the repository is organised and how to run the code.

| | |
|---|---|
| Data | 15,358 15-minute windows ≈ 3,840 hours, 3 observatories |
| Type III | **92%** detection, **5.0 s** median timing error, F1 **0.57** |
| Type V (after adding a second station) | detection **4% → 55%** |
| Zero-shot cross-instrument transfer | **77%** detection, no fine-tuning |
| Events missing from the reference catalog | **126** found by the model and confirmed by review |

---

## Documentation map

Four categories. **New readers: start with `RESULTS.md`, then `TRAINING_PLAN.md`,
then the HANDOFF of whichever module you are touching.**

> Note: `README.md` and `RESULTS.md` are in English. The working documents below
> are in Chinese — they are lab notebooks kept in the language they were written
> in, and `RESULTS.md` distils everything load-bearing out of them.

### Results and design (repository root)

| File | Contents |
|---|---|
| **[`RESULTS.md`](RESULTS.md)** | **Consolidated results: every number, the falsified claims, the methodological traps. Read this first** |
| [`TRAINING_PLAN.md`](TRAINING_PLAN.md) | Design rationale: why detection rather than classification, how bounding boxes are defined, phase plan, literature basis |
| [`PHASE1_RESULTS.md`](PHASE1_RESULTS.md) | The **full experimental log** from Phase 1 through Phase 3, with raw numbers and the judgement made at each point. `RESULTS.md` is its distillation |
| [`HANDOFF.md`](HANDOFF.md) | Early data-processing handoff: denoising and scraping history for own-station and e-Callisto |
| [`OVERNIGHT_PLAN.md`](OVERNIGHT_PLAN.md) | Design and execution records for the unattended overnight batches |

### Module documentation

| File | Contents |
|---|---|
| [`event_review/HANDOFF.md`](event_review/HANDOFF.md) | Review tool design, code structure, and how the two review subsets are constructed |
| [`event_review/README.md`](event_review/README.md) | Review tool quick start |
| [`ecallisto_grabber/README.md`](ecallisto_grabber/README.md) | Scraper usage |
| [`ecallisto_grabber/开发日志.md`](ecallisto_grabber/开发日志.md) | Scraping and denoising development log (bugs found on real data) |

### Superseded

The classification + GAN-augmentation approach that predates the switch to
object detection. Kept for reference; not part of the current pipeline.

- `dcgan/` — DCGAN / SpecGAN data augmentation
- `radburst_tl/` — early transfer-learning training scripts

---

## Layout

```
detection/          main pipeline: dataset construction, training, evaluation, gap mining
transfer/           own-station transfer experiments (Phase 3)
event_review/       human review tool (Streamlit) + review-subset generation
ecallisto_grabber/  e-Callisto scraping + sumthreshold denoising
data/               spectrogram arrays and annotations (mostly untracked, see .gitignore)
```

Only **hand-made, irreproducible** CSVs under `data/` are version-controlled
(review records, catalog-gap boxes, the contaminated-negative list). Multi-GB
`.npy` files and regenerable YOLO dataset directories are not; the reasoning is
written at the top of [`.gitignore`](.gitignore).

---

## Pipeline

### 1. Scrape

```bash
python ecallisto_grabber/scrape_windows.py \
    --start 2021-01-01 --end 2024-03-31 --stations Arecibo-Observatory \
    --types II III V --out-dir data/ecallisto/windows
```

Downloads whole station-days, cuts 15-minute windows on FITS boundaries, saves
`.npy`, and is resumable. **Note that `--limit` is a global budget, not a
per-station allocation** — multi-station runs must be issued one station at a
time.

### 2. Human review

```bash
streamlit run event_review/app.py
```

Pick the source and directory in the sidebar. The core feature is **dragging
horizontally across the spectrogram to set burst start and end directly** —
catalog times are systematically early (measured median +40 s), and correcting
that is the main job.

Targeted re-review subsets:

```bash
python event_review/make_box_subset.py data/ecallisto/windows_assa \
    --types II V --out-dir data/ecallisto/assa_iiv --apply
```

### 3. Build a dataset and train

```bash
python detection/build_yolo_dataset.py data/ecallisto/windows data/ecallisto/yolo8_single \
    --train-source clean --min-window-frac 0.99 --single-class \
    --contaminated-negatives-scope train --stations Arecibo-Observatory

python detection/train_yolo.py data/ecallisto/yolo8_single/data.yaml \
    --name run1 --model yolov8s.pt --freeze-epochs 5 --epochs 30 --patience 10
```

Three deliberate non-default decisions (read the script docstrings before
changing them): **split by station-day, never by window**; **validation accepts
only human-reviewed clean windows**; **all geometric augmentation is off**
(flipping a spectrogram destroys frequency drift direction, the primary Type
II/III cue).

⚠️ **Run ≥3 seeds.** The noise floor σ is somewhere between 0.018 and 0.056, and
a single run is uninterpretable.

⚠️ With `--optimizer` left at its default `auto`, **ultralytics overrides
`--lr0-*`** and the learning rate silently has no effect (a startup warning now
fires). Searching learning rates requires naming an optimiser explicitly.

### 4. Evaluate

```bash
python detection/evaluate.py detection/runs/run1/weights/best.pt \
    data/ecallisto/yolo8_single/data.yaml --imgsz 640 --conf 0.05 --maxconf
```

Reports two views: standard AP@0.30/0.50/0.75, plus **operational metrics**
(detection rate, timing error, false alarms per hour). AP collapses "missed it
entirely" and "found it, 21 s off" into the same zero, and those are very
different outcomes for a user.

**Pin the protocol across every comparison** — both `--rect` and `--maxconf`
move the numbers substantially.

### 5. Use the model to find events the catalog missed

```bash
python detection/mine_catalog_gaps.py detection/runs/run1/weights/best.pt \
    data/ecallisto/windows detection/gap_review --min-conf 0.10
# after a human fills the verdict column in gap_candidates.csv:
python detection/apply_gap_verdicts.py detection/gap_review/gap_candidates.csv --apply
python event_review/make_gap_subset.py --apply          # draw the boxes in the review tool
python detection/apply_gap_annotations.py --type-override gap0067=II --apply
```

The reference catalog omits roughly 36% of events, and this loop is the fastest
route to more clean labels. **Model-drawn boxes never enter training directly** —
they serve only as the prefill, and the time range is redrawn by hand.

### 6. Own-station transfer

```bash
python transfer/make_own_windows.py --apply        # continuous CSV -> real 15-minute windows
python transfer/zero_shot_transfer.py detection/runs/run1/weights/best.pt
```

Compares two renderings: physically correct band placement versus a naive
stretch. **Measured: the naive stretch wins by a wide margin** (77% vs 14%) — see
`RESULTS.md`.

---

## Environment

- Tesla T4 (15 GB), PyTorch 2.12 + CUDA, ultralytics 8.4.80
- The machine has only **4 CPU cores, 15 GB RAM, and no swap**. Training defaults
  to 8 dataloader workers, which oversubscribes it; use `--workers 2` when
  sharing the machine. The review tool holds about 4 GB — do not leave it running
  during training (it has been OOM-killed twice).

## One trap worth repeating

`pgrep -f <script name>` and `pkill -f <script name>` **match the command line of
the shell that invoked them**. This cost 4 hours once (a waiter script that
deadlocked against itself) and killed the wrong process twice more. Use a port
(`ss -lptn 'sport = :8501'`) or a more specific pattern for any wait or cleanup
logic — never the script name alone.
