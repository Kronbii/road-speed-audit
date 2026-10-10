# Portfolio summary: road-speed-audit

All measurements below are from **simulated** video (the repository's own
generator, with exact ground truth). No field validation has been done yet. Each
number names the results file it comes from.

## One-paragraph summary

`road-speed-audit` measures vehicle speeds on one road segment from one fixed
roadside camera, offline and without storing images of people or number plates.
A homography calibrated from a handful of tape-surveyed road points maps the image
to metres; a background-subtraction detector with sub-pixel bottom-edge refinement,
a ByteTrack-style tracker with merged-blob handling and road-coordinate track
stitching follow each vehicle; a robust position-versus-time fit over a measurement
zone gives its speed, a Monte-Carlo uncertainty and quality flags; doubtful records
go to a human review queue; and a report gives a municipality mean speed, V85,
share over the limit and flow with confidence intervals. To find out how far the
numbers can be trusted before any field work, I built a numpy/OpenCV renderer of
road scenes with a calibrated pinhole camera and ran 221 simulated runs across 13
experiments: on the baseline set-up 88 % of simulated vehicles got a valid speed
with a 95th-percentile error of 1.01 km/h, and the sweeps turn into concrete,
caveated placement guidance for the first municipal pilot.

## Headline measured results (simulated)

1. **Baseline accuracy** — 118 of 134 simulated vehicles (88 %) received a valid
   speed; bias −0.23 km/h, RMSE 0.52 km/h, 95th percentile of |error| 1.01 km/h
   (95 % CI 0.67–1.85), no false or duplicate valid records
   (`results/e01_baseline.json`).
2. **V85 for a municipality** — over 20 simulated traffic runs, V85 RMSE 1.22 km/h
   and mean-speed RMSE 0.60 km/h against the truth; all 20 reported 95 % intervals
   contained the true V85 (`results/e10_v85.json`).
3. **Distance is the main limit** — 95th percentile of |error| 0.76 km/h with the
   zone centred 25 m from the pole, 3.59 km/h at 55 m, 10.84 km/h at 95 m
   (`results/e02_distance.json`).
4. **Calibration layout beats tape precision** — 95th-percentile calibration-induced
   bias 7.08 % of speed with four bunched points vs 1.32 % with nine points over the
   zone; survey noise up to 10 cm per axis moved it only from 1.27 % to 1.53 %
   (`results/e09_calibration.json`).
5. **Sub-pixel bottom edge matters** — the raw blob boundary biased speeds by
   −1.30 km/h; the refined edge by −0.10 km/h (`results/e12_contact.json`).

## CV lines

- Built an offline camera-based vehicle-speed audit tool (homography calibration,
  background subtraction, Kalman/Hungarian tracker, robust speed fit with
  Monte-Carlo uncertainty, human review queue, municipal V85 report) in Python;
  on simulated video with exact ground truth it measured 88 % of vehicles with a
  95th-percentile speed error of 1.0 km/h.
- Wrote a numpy/OpenCV road-scene simulator (calibrated pinhole camera, 3-D vehicle
  models, shadows, blur, noise, occluders, ID-buffer ground truth) and ran 221
  seeded runs across 13 experiments to derive camera-placement guidance; found that
  a compact calibration layout causes about 5× the speed bias of points spread over
  the zone.
- Designed the privacy and validation side: numbers-only outputs, anonymised review
  crops, an ethics statement, and a GNSS/radar field protocol with Bland–Altman
  acceptance criteria; CI runs lint, 58 tests and a simulated end-to-end check.

## Suggested portfolio fields

**What.** An open-source, offline tool that measures vehicle speeds on one road
segment from one fixed camera and reports the speed statistics a municipality
uses, with uncertainty and human review; plus a simulator that tests it against
exact ground truth.

**Problem.** Speed drives crash severity (WHO: every 1% increase in mean speed
produces a 4% increase in fatal crash risk), and Lebanon's WHO 2023 country profile
shows a 50 km/h urban limit that local authorities can modify; but a municipality
rarely has speed data for a specific street, and cheap camera measurements are only
useful if their accuracy is known and they do not become surveillance.

**How.** Tape-surveyed homography; MOG2 background subtraction with shadow-aware
masks and sub-pixel bottom-edge refinement; SORT/ByteTrack-style tracking with
merged-blob handling; stitching of partial tracks in road coordinates; Huber line
fit over a measurement zone; Monte-Carlo propagation of calibration and pixel noise
carried into bootstrap confidence intervals for mean speed and V85; numbers-only
outputs and a review queue; validated with a purpose-built simulator over 13
experiments.

**Role.** Designed and implemented the tool, the simulator, the experiments and the
documentation for this repository.

**Stages.**

| step | title | detail |
|---|---|---|
| 1 | Geometry and calibration | Normalised DLT homography with image-space refinement, pinhole camera recovery, lane-marking helper, interactive picker |
| 2 | Detect, track, measure | MOG2 with sub-pixel bottom edge, two-stage Hungarian tracker with merged-blob handling, track stitching, robust speed fit with quality flags |
| 3 | Uncertainty and reporting | Monte-Carlo calibration and pixel noise, review queue with anonymised sheets, HTML report with V85 and intervals |
| 4 | Simulated validation | numpy/OpenCV renderer with exact ground truth; 221 runs, 13 experiments; placement guidance |
| 5 | Field-pilot preparation | GNSS/radar protocol with Bland–Altman acceptance criteria, ethics and retention rules |

**Measurements.**

| label | value | context |
|---|---|---|
| Valid speeds, baseline | 88 % (118 / 134) | simulated, 2-lane road, 6 m pole, 720p, 30 fps, zone 20–50 m (`e01_baseline.json`) |
| 95th-percentile speed error | 1.01 km/h (95 % CI 0.67–1.85) | same baseline, valid records |
| V85 error | RMSE 1.22 km/h | 20 simulated traffic runs, 19–29 vehicles each (`e10_v85.json`) |
| Error vs distance | 0.76 → 3.59 km/h | 95th percentile, zone centre 25 m vs 55 m from the pole (`e02_distance.json`) |
| Calibration layout | 7.08 % vs 1.32 % | 95th-percentile calibration-induced bias, bunched vs spread points (`e09_calibration.json`) |

**Limits.** Simulation only: flat straight road, ideal lens, box-shaped vehicles,
constant speeds, daylight; the method and simulator were developed together. The
error rises steeply with distance and road length per pixel, a single lamp post in
the line of sight halved the valid share, and without cast shadows the bias doubled
and the stated intervals became too narrow. Not certified and not for enforcement;
field validation against GNSS/radar is the next step.

**Hero image.** `figures/hero_frame.png` (simulated scene with measured vs true
speeds, the measurement zone and contact-point trails; labelled "SIMULATED SCENE").
Secondary: `figures/e02_distance.png` (error and valid share vs distance) or
`figures/e09_calibration.png`.
