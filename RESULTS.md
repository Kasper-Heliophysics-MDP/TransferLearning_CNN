# Solar Radio Burst Detection — Results

Consolidated from `PHASE1_RESULTS.md`, `TRAINING_PLAN.md`, `OVERNIGHT_PLAN.md` and
`HANDOFF.md`. This file covers **what was built, what the numbers are, and which
conclusions were overturned along the way**. Process detail and debugging history
stay in the original documents — see `README.md` for the map.

Data as of 2026-08-16.

---

## In one sentence

A COCO-pretrained YOLOv8 detector (not a classifier) over 15-minute spectrogram
windows reaches **92% Type III detection at 5.0s median timing error** across
3 observatories and 3,840 hours of data, and **transfers zero-shot to a
completely different receiver at 77% detection**.

Two by-products turned out to matter as much as the detector: the accuracy
ceiling was shown to be **label quality rather than model capacity**, and the
trained detector was used in reverse to **recover 126 real events the reference
catalog never recorded**.

---

## Scale

| | |
|---|---|
| Spectrogram windows | **15,358** 15-minute windows ≈ **3,840 hours** |
| Observatories | Arecibo-Observatory, Australia-ASSA, own station (3 Michigan sites) |
| Catalog boxes | 4,336 (e-Callisto) + 65 (own station) |
| Boxes found by the model and confirmed by review | **126** (absent from the catalog) |
| Total human judgements | **≈ 1,650** |

---

## Main results

### 1. Detection performance (Type III, single-class detector)

Arecibo, 243 validation boxes, 4 random seeds.

| Metric | Best seed | 4-seed mean |
|---|---|---|
| AP@0.50 | 0.417 | 0.402 (σ 0.018) |
| **Detection rate** | 93% | **92%** |
| **Median timing error** | **5.0 s** | 5.0–5.9 s |
| **F1** (maxconf operating point) | **0.566** | — |

**AP and the operational metrics tell different stories and have to be read
together.** AP@0.50 = 0.40 looks far below the ~0.80 reported in the literature,
yet the same model finds 92% of the bursts with a 5-second median localisation
error. The reason is the next section.

`maxconf` is the deployable read (keep only the highest-confidence box per
cluster). It raises F1 from 0.335 to **0.566** and drops false alarms from
1.43/h to 0.51/h. The cost is slightly worse localisation (|Δt| 5.0 → 8.5 s):
it keeps the most *confident* box in a cluster, not the best-placed one.

### 2. The ceiling is the labels, not the model — the key methodological result

Boxes span the full frequency height, so IoU is determined entirely by the time
axis: two equal-width intervals offset by `d` have `IoU=(w−d)/(w+d)`, and
therefore **IoU≥0.5 requires `d ≤ w/3`** — only 23 seconds for a median 69s box,
and 10 seconds for a 30s one.

The measured median timing error of the reference catalog (Monstein) is
**+40 seconds**. That gives:

| | Upper bound |
|---|---|
| IoU≥0.50 hit rate (= the AP50 ceiling) | **48.5%** |
| IoU≥0.75 | 12.4% |
| **IoU≥0.95** | **0.0%** |

**This is why AP50-95 never moved across four rounds of experiments
(0.034–0.058)** — that metric was pinned by the labels from the start,
independent of the model.

**Verified with a falsifiable prediction written down in advance:**
**205 hand-corrected boxes (AP50 0.269) beat 1,500 mixed boxes (0.184)** with
7.3× less training data. At this scale, adding unreviewed labels to the training
set is a *negative* contribution, not merely a neutral one.

Human annotation repeatability was measured at **88%** (25 events blind
re-annotated, median IoU 1.000), so the ceiling can be lifted from 0.485 to
~0.88. That makes expanding manual annotation a known-good investment rather
than a gamble.

### 3. The reference catalog omits ~36% of real events

34 "false positives" were checked by eye one by one: **32 (94%) were real
bursts**, and all 24 with confidence ≥0.1 were real without exception.

Three reported numbers had to change: false-alarm rate 0.90/h → **~0.05/h**;
precision severely underestimated; catalog completeness from "assumed 100%" to
**~36% omitted**.

**The worse consequence is that training received active wrong supervision.**
Windows holding a real burst but used as negatives (empty labels) were teaching
the model "no burst here". That is a reversed signal, not just label noise.

**What was built from this:** `mine_catalog_gaps.py` runs the detector over every
window and lists high-confidence detections with no catalog entry, so a human
only has to confirm them. 160 candidates, 133 reviewed, **126 confirmed real**.
The confirmation rate rises monotonically with confidence (89% / 94% / 100% /
100%), landing on the same curve as the independent 34-sample check made three
weeks earlier.

Side effect: of the 90 windows previously discarded as "negatives known to hide a
burst", **63 now carry real labels** and rejoin training as ordinary positives —
turning "throw away 90 windows" into "throw away 27, recover 63".

### 4. A second station rescues Type V

Type V was the one outright failure of Phase 1: Arecibo has only 27 clean samples
in total, which cannot feed training and evaluation at once (a 20-box validation
set leaves 7 to train on). Australia-ASSA was scraped for this reason, and its
155 II/V boxes were reviewed by hand (149 usable, 6 discarded, 136 with corrected
times).

The two arms are **single-variable**: validation is byte-identical (1,096 images /
287 boxes), **Type III training boxes are held at 582 in both arms**, and only II
(50→105) and V (14→34) change. 3 seeds per arm.

| | Arecibo only | +ASSA | |
|---|---|---|---|
| **Type V detection** | **4%** (0–12%) | **55%** (48–70%) | |
| Type V AP@0.50 | 0.006 | **0.208** | ranges do not overlap |
| Type II detection | 44% | **78%** | |
| Type II AP@0.50 | 0.173 | 0.300 | |
| Type III AP@0.50 | 0.292 | 0.339 | |

Two of the three Arecibo-only seeds detected **no Type V at all**. With ASSA, all
three seeds land in 0.147–0.250. This needs no appeal to σ — the ranges are
disjoint.

**An unplanned effect that may matter more than the AP numbers: training
stability.**

| Seed-to-seed spread | Arecibo only | +ASSA |
|---|---|---|
| Type II detection | **68 pt** (0–68%) | 9 pt |
| Type III detection | **44 pt** (49–93%) | **1 pt** (91–92%) |

One Arecibo-only seed collapsed entirely (II and V at zero, III down to 49%).
With ASSA the collapse stops happening — **and Type III's own training boxes were
identical between arms**, so III improved purely because the other two detection
heads stopped starving the shared backbone.

**At this data scale the real failure mode is not score jitter, it is a training
run dying outright**, which a single-seed experiment cannot detect.

### 5. Zero-shot cross-instrument transfer works

The own-station receiver differs sharply from the training data:

| | Own station | Training data | Ratio |
|---|---|---|---|
| Band | 15.996–24.004 MHz | 15–86.6 MHz | 8.9× |
| Channel spacing | 0.0195 MHz | 0.358 MHz | **18.4×** |
| Sample interval | 0.1 s | 0.25 s | 2.5× |

**62 real 15-minute windows** were cut from 49 continuous raw CSVs on absolute
time boundaries — not cropped around events. Box-centre position has
std = **0.294**, versus 0.0000 for event-centred crops, which is exactly the
label leak that invalidated the first Phase 1 attempt.

With **no fine-tuning at all**, using the e-Callisto-trained model:

| | Detection rate | Median \|Δt\| |
|---|---|---|
| **Zero-shot transfer** | **77%** (50/65) | **7.5 s** |
| Permutation null (200 shuffles) | mean 25%, 95th pct 32% | — |

The real pairing sits **12.2 standard deviations** above the permutation
distribution, ruling out the explanation that the model simply sprays boxes over
RFI-saturated images and hits by chance.

Reviewing the 35 unmatched detections: **20 are real bursts**. Therefore:

- True false-alarm rate **0.97/h** (the reported 2.4/h was only an upper bound)
- The **own-station catalog omits ~24%** of events (36% measured on e-Callisto)
- Recall against *all* real bursts is roughly **82%**

### 6. Frequency alignment — the "biggest open problem" — turned out not to need solving

`TRAINING_PLAN.md` treated cross-station frequency alignment as the main blocker
and planned to resolve it by interpolating onto a common frequency grid. Testing
both renderings directly settled it:

| Rendering | Detection rate | Max confidence |
|---|---|---|
| **Physically correct** (8 MHz placed at its true 72 of 640 rows) | **14%** | 0.351 |
| **Physically wrong** (8 MHz stretched to full height) | **77%** | 0.962 |

**The physically correct one nearly fails completely.** The model keys on the
*appearance* of a burst, not on absolute physical drift slope. Stretching 8 MHz
to full height makes the slope wrong by 8.9× but makes the image look like the
training data; compressing it into a 72-pixel strip on a blank canvas is
physically exact but unlike anything the model has seen.

**Conclusion: Phase 3 does not require frequency alignment.** This also explains
why the multi-station papers surveyed never mention handling it — they may
genuinely not need to.

---

## Claims that were falsified or overturned

Kept on the record because they save the next person from repeating the work.

| Earlier belief | Measured outcome |
|---|---|
| Adding a frequency range to boxes would relax IoU | **Gain is identically zero.** The frequency term cancels in the 2-D IoU numerator and denominator |
| 1-D interval IoU is stricter than 2-D boxes | **Backwards.** The full-height convention is in fact *more* permissive |
| Automatic time correction could replace manual work | **All 10 settings lost badly to doing nothing** (IoU≥0.5: 9–32% vs 56%) |
| Higher time resolution (640×1280) would improve localisation | **Harmful, not merely unhelpful.** Two runs (batch 8 / 16) gave 0.130 / 0.071; the second peaked at epoch 1 and never improved |
| Cross-station frequency alignment is a prerequisite | **Not needed** — see above |
| Denoising own-station data would improve transfer | **It does not** — see below |
| Single-class beats 3-class by 4× | **Overstated.** A fair comparison (both at 570 boxes) is +19%, and the multi-class model had better detection rate and localisation |
| "All four arms differ only by noise" | **Also overstated.** Seed noise was smaller than the four-arm spread; only A/D were unreadable |

### The negative result on own-station denoising

The station has a full sumthreshold denoising pipeline, but **the production
parameters cut transfer detection from 77% to 40%** and degrade timing error from
7.5s to 60s.

| Input | Detection rate | Median \|Δt\| |
|---|---|---|
| **raw (no denoising)** | **77%** | **7.5 s** |
| blind, `vrfi_cov=0.95` | 77% | 15.0 s |
| blind, `vrfi_cov=0.60` | 66% | 23.2 s |
| blind, `vrfi_cov=0.10` (production default) | **40%** | 59.6 s |

**The cause is structural, not a tuning failure.** Within an 8 MHz band a Type
III burst *is* a full-height vertical stripe — which is precisely
`vertical_rfi_weight`'s criterion for RFI. The two are not separable there.

The parameter sweep confirms it: `base_threshold_sigma` from 12 to 40 barely
moves burst retention (0.08 → 0.09), while `vrfi_coverage_threshold` from 0.10 to
0.95 moves it from **0.08 to 0.73**.

Verified by eye as required: across the 12 worst cases, every burst is a
saturated band in the raw panel and blank in the denoised one.

**This also explains an older observation.** "Burst looks faint / was erased"
came up repeatedly during review and was attributed to `cleaned_events/` having
been generated with stale timing. For own-station data there is a second,
independent cause: **even with perfectly correct timing, a narrow band makes
vertical-RFI suppression damage the burst**. It is also why `known_burst_weight`
is a requirement rather than an optimisation here.

Making this path work needs a different algorithm — for example using persistence
along the time axis (RFI tends to recur at the same frequency across many
windows; bursts do not) — not parameter tuning.

---

## Three things to know about the method

### The noise floor is larger than expected and hard to pin down

Standard deviation of AP50 across seeds at a fixed configuration:

| Measurement | σ | 95% interval (n=4) |
|---|---|---|
| 2026-08-14 (570 training boxes) | 0.056 | [0.032, 0.209] |
| 2026-08-15 (648 training boxes) | 0.018 | [0.010, 0.066] |

**The two intervals overlap, so no reduction in the noise floor can be claimed.**
Estimating σ from n=4 is simply very imprecise. In practice: **run ≥3 seeds for
any configuration; a single run is uninterpretable**, and treat any difference
below roughly 2σ as no difference.

Historical lesson: skipping this once led to reading jitter from the `rect`
evaluation flag as a real effect and declaring a winner, later retracted.

### The evaluation protocol has to be pinned

`rect` is an ultralytics `val()` flag that moves AP50 for the **same weights**
from 0.176 to 0.203. Every comparison must pass the same value. The same applies
to `--maxconf`, which lifts F1 from 0.335 to 0.566 — the two readings are far
apart, so any quoted figure must say which one it is.

### Dataset construction can create label leakage silently

Crops cut as "event ± fixed buffer" put the box centre at exactly the middle of
the image every time (measured std = 0.0000, n = 1694). Any mAP computed on such
data is uninterpretable — it cannot distinguish "learned to find bursts" from
"learned to draw a box in the middle". **Check the distribution of label
positions before any detection or localisation work**; it is a one-line test. The
same check was applied when building the own-station windows (std = 0.294).

---

## What has not been done

Listed explicitly so the results are not read as covering more than they do.

- **No held-out test set.** `best.pt` is selected by early stopping on the
  validation set, so model selection and reporting share data; the numbers are
  mildly optimistic.
- **Own-station transfer is zero-shot only** — no fine-tuning was attempted.
  Fine-tuning on part of the 62 windows would likely improve it.
- **Type III AP remains 0.33–0.42.** The labelling ceiling was located and
  explained, not removed.
- **ASSA's 2,092 Type III boxes are unreviewed** and therefore excluded from
  training (mixing unreviewed labels is a measured negative).
- **No gap mining has been run on ASSA**, so its negative windows may contain
  unlabelled bursts; they were excluded from Phase 2 entirely.
- **Own-station Type II/V samples are too few** (9 and 2), so the transfer result
  effectively covers Type III only.
- **Own-station denoising is unsolved**; transfer currently runs on raw data.
- The earlier DCGAN / SpecGAN augmentation branch (`dcgan/`, `radburst_tl/`)
  predates the switch to a detection architecture and is not part of the current
  pipeline.
