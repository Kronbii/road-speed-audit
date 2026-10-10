# Field validation protocol (draft for the first pilot)

**Status: this protocol has not been carried out.** Every accuracy figure in this
repository so far comes from simulated video. Before any speed statistic from a
real site is published, the installed camera must be validated against an
independent reference at that site, following a protocol like this one, agreed
with the municipality in advance.

The aim is narrow: for **one camera at one road segment**, estimate how far the
tool's per-vehicle speeds are from a trusted reference (bias and limits of
agreement), whether its stated uncertainty is honest, and whether it misses or
double-counts vehicles. Acceptance criteria are fixed *before* the data are
collected.

---

## 1. Roles and permissions

- A written agreement with the municipality covering purpose (audit statistics,
  not enforcement), the site, dates, who holds the raw video, retention and
  deletion, and how results will be published. See [ethics.md](ethics.md).
- Check obligations under Lebanese Law No. 81 of 2018 on Electronic Transactions
  and Personal Data ([UN ESCWA legislation database](https://alp.unescwa.org/legislations/law-81-2018-electronic-transactions-and-personal-data))
  with the municipality's legal adviser before recording.
- Coordination with the municipal police for any work on the carriageway and for
  probe-vehicle runs. Field staff wear high-visibility clothing; nobody stands on
  the carriageway without traffic management.
- Public notice at the site that a camera is measuring traffic speeds for a road
  safety study, with a contact.

## 2. Site selection and survey

**Site.** A straight segment with a flat road surface over the measurement zone
and at least 25 m before and 60 m beyond it (the region the tracker uses), a
mounting point (lamp post, building) on the roadside, and no regular parking in
the zone. The homography assumes the road is a plane: note crests, dips and
strong camber; if the surface deviates from a plane by more than about 10 cm over
the zone, choose another site or shorten the zone.

**Starting point for the camera set-up** (from simulation, to be confirmed by this
protocol — see `results/guidance.json` and the README): mount the camera at least
4.5 m high, keep the zone centre within about 25–40 m of the pole so that one pixel
covers at most about 33 cm of road along the road at the zone centre, aim at the
zone centre, keep poles and trees out of the line of sight to the zone, and keep
the zone fully inside the image with margin. Frame rates of 10–50 frames/s made no
difference in simulation; record at the camera's native rate. The calibration
points must surround the zone.

**Ground control points.** At least 8 (minimum 6) points on the road surface that
are visible in the image and can be located to a centimetre: corners of dashed
lane markings, painted dots, kerb joints. Spread them over the whole zone and
across the full road width (both edges and the centre line, at the start, middle
and end of the zone). Avoid points only at one end of the zone; simulation shows
that a compact layout multiplies calibration error (`results/e09_calibration.json`).

**Coordinates with a tape (no total station needed).**

1. Lay a baseline A–B along the road edge, roughly the length of the zone, and
   mark A and B. Define the local frame: origin at A, +Y from A towards B (along
   the road, away from the camera if possible), +X across the road to the right
   when looking along +Y, so that the frame is right-handed with Z up.
2. For every control point P, measure the distances A–P and B–P with a steel or
   fibreglass tape (30–50 m), pulled taut and level. Compute P by trilateration
   (two circles; take the solution on the road side of the baseline).
3. Measure every distance twice, by two different people. The difference between
   the two readings estimates the survey error; use its standard deviation as
   `ground_sigma_m` in the site configuration. Remeasure any pair that differs by
   more than 3 cm.
4. Measure lane widths and the dash length and gap at several places as
   independent checks (and for the lane-marking calibration helper).
5. Record the tape's identity, temperature and the names of the measurers.

**Check distances.** Mark two or three pairs of points that are *not* used for
calibration (e.g. dash ends 12–30 m apart). After calibration, their image
positions mapped to the ground must reproduce the taped distances; report the
percentage error. This is the first, cheapest test of the calibration.

## 3. Calibration

1. Record a short clip with no traffic in the zone, or extract a frame in a gap.
2. `roadspeed calibrate --video clip.mp4 --frame N --survey survey.json --out site.json --zone START END --lane ... --speed-limit L --check-image check.png`
   (click each surveyed point in the order listed; a magnifier helps to place the
   click to a pixel).
3. Accept the calibration only if the reprojection RMS is small (aim for ≤ 1.5 px;
   inspect any point with a residual above 3 px), the check distances agree, and
   the overlay (`check.png`) shows lane lines and zone on the road where expected.
   The overlay contains a frame of the scene: keep it with the raw data, not in
   any published material.
4. Freeze `site.json` (record its SHA-256) before the reference measurements start.

## 4. Reference measurements

Use one or both methods. Synchronise all clocks before starting: film an
NTP-synchronised clock display at the beginning and end of each session (point the
camera at the display only, no people in view), and log GNSS time on the probe.

### 4a. GNSS probe vehicle (primary)

- A car with a GNSS receiver that logs Doppler speed and position at ≥ 10 Hz,
  antenna on the roof. Note the receiver model and its stated speed accuracy.
- Drive through the zone at steady speeds using cruise control where possible:
  for example 30, 40, 50 km/h and, only if the speed limit and police permit it
  (e.g. on a temporarily closed road), higher speeds. Agreement is only
  established for the speeds actually tested.
- Cover every lane and both directions, at different times of day if possible
  (sun angle changes shadows, which the simulation shows can matter).
- **Number of passes.** At least 30 valid passes per direction (60 in total);
  aim for 50 per direction. With *n* passes the 95 % limits of agreement are
  themselves uncertain by about ±1.96·√(3/n)·SD (Bland & Altman, 1986), i.e.
  ±0.62·SD for n = 30 and ±0.48·SD for n = 50.
- Reference speed per pass: the GNSS mean speed over the zone, computed from the
  logged positions (or Doppler speeds) between the times the vehicle enters and
  leaves the zone.

### 4b. Radar or lidar speed gun (secondary, ordinary traffic)

- A calibrated device (calibration certificate; tuning-fork or self-test check at
  start and end). Position it to keep the angle to the direction of travel small;
  the cosine effect makes radar read low by a factor cos θ (−1.5 % at 10°).
- An operator records time, lane, direction and speed for free-flowing vehicles
  inside the zone, ideally ≥ 100 vehicles. Match to camera records by time stamp
  and lane; discard ambiguous matches (two vehicles within 2 s in the same lane).
- Nobody photographs drivers or plates; the log contains time, lane, direction and
  speed only.

### 4c. Counts

For at least four 15-minute periods, two people independently count vehicles
crossing the zone per lane (on site or from the raw video, which stays on the
offline machine). Compare with the tool's counted records to estimate missed and
duplicate records.

## 5. Analysis

For each matched pass: difference *d* = camera speed − reference speed.

1. **Bland–Altman plot** (Bland & Altman, 1986): *d* against the mean of the two
   speeds. Report the bias (mean *d*) with its 95 % CI, the 95 % limits of
   agreement (bias ± 1.96·SD) with their CIs, separately per direction and lane
   and pooled.
2. **Proportional bias:** regress *d* on the mean speed. A slope different from
   zero indicates a scale error (calibration, reference-point drift). In simulation
   (`results/e13_probe.json`) a scale factor estimated from five known-speed passes
   halved the small residual bias of the 720p baseline; whether that carries over
   must be tested here by estimating the factor on half of the passes and checking
   it on the other half.
3. **Honesty of stated uncertainty:** the share of passes with |*d*| ≤ 1.96·√(u² + u_ref²),
   where u is the tool's per-vehicle uncertainty and u_ref the reference's; it
   should be close to 95 %.
4. **Completeness:** missed vehicles, duplicate records and false records from the
   counts in 4c, and the share of flagged records sent to review.
5. **Acceptance (proposed; fix before data collection):** for audit statistics,
   |bias| ≤ 1 km/h, 95 % limits of agreement within ±3 km/h, uncertainty coverage
   between 90 % and 98 %, and at least 80 % of counted vehicles with a valid speed.
   If the site fails, do not publish speed statistics from it; change the set-up
   and validate again.

## 6. Data handling and retention

- Raw video is recorded to an encrypted disk on an offline computer, is never
  uploaded, and is accessible only to named team members. It is deleted after the
  validation analysis and the human review of flagged records are complete, and in
  any case no later than 30 days after recording (or a shorter period agreed with
  the municipality). Deletion is logged.
- Kept long term: the records (time, lane, direction, speed, uncertainty, flags),
  the site configuration and calibration data, the reference logs (speeds and
  times only), and the analysis outputs.
- Review sheets (`--review-sheets`) contain pixelated, blurred crops only and are
  deleted with the raw video.
- Anything published is aggregated (see [ethics.md](ethics.md)).

## 7. Validation report

Publish with the results: site description (no imagery of people), dates,
weather and light, camera and lens, software version and `site.json` hash,
calibration residuals and check distances, number of passes per lane/direction
and speed, the Bland–Altman figures and numbers with CIs, completeness, the
acceptance decision, and the known limitations (untested speeds, night, rain).

## Not covered

Type approval or certification of a measuring instrument; enforcement; night and
rain operation (to be validated separately if needed); curved or sloped roads.

## References

- Bland, J. M., & Altman, D. G. (1986). Statistical methods for assessing agreement
  between two methods of clinical measurement. *The Lancet*, 327(8476), 307–310.
- World Health Organization (2023). *Global status report on road safety 2023:
  country profile, Lebanon.* https://www.who.int/publications/m/item/road-safety-lbn-2023-country-profile
