# Calibration guide

The tool maps image pixels to metres on the road with a **homography** estimated
from at least four image ↔ ground point pairs. Use at least six (better eight to
ten) so the residuals mean something; with exactly four the fit is exact and the
residuals are zero whatever the errors.

## Ground frame

Metres on the road surface, right-handed with Z up:

- `X` across the road, increasing to the right when looking along +Y;
- `Y` along the road, increasing away from the camera (recommended).

The default road axis is `{"origin": [0, 0], "direction": [0, 1]}`: speeds are
measured along +Y, "away" means increasing Y. Lanes are given by their lateral
centre `center_m` (in X) and width.

## Three ways to calibrate

### 1. Surveyed points, clicked interactively (recommended)

Survey the points with a tape (see [field-protocol.md](field-protocol.md)) and
write them to `survey.json`:

```json
[
  {"label": "P1", "ground": [-3.5, 20.0]},
  {"label": "P2", "ground": [0.0, 20.0]},
  {"label": "P3", "ground": [3.5, 20.0]}
]
```

Then click them in order on a frame of your video:

```bash
roadspeed calibrate --video clip.mp4 --frame 120 --survey survey.json \
  --out site.json --zone 20 50 \
  --lane towards-1:-1.75:3.5:towards --lane away-1:1.75:3.5:away \
  --speed-limit 50 --click-sigma 1.0 --ground-sigma 0.02 \
  --check-image check.png
```

Keys in the picker: left click adds a point, `u` undoes, `Esc` aborts; a
magnifier in the corner shows the pixels under the cursor. Without `--survey` the
terminal asks for the ground coordinates after each click.

### 2. Point pairs from a file

```json
[{"image": [612.0, 455.0], "ground": [-3.5, 20.0]}, ...]
```

```bash
roadspeed calibrate --points pairs.json --size 1280 720 --out site.json --zone 20 50 ...
```

### 3. Lane-marking geometry (helper)

When the dashed lane lines have a known, *measured* geometry (lane width, dash
length and gap), click dash ends instead of surveying every point:

```json
{
  "dash_length_m": 3.0,
  "dash_gap_m": 9.0,
  "lines": [
    {"x_m": 0.0, "phase_m": 0.0,
     "points": [{"uv": [640, 500], "dash": 0, "end": "near"},
                {"uv": [652, 470], "dash": 0, "end": "far"},
                {"uv": [688, 380], "dash": 2, "end": "near"}]},
    {"x_m": 3.5, "phase_m": 0.0, "points": [ ... ]}
  ]
}
```

`dash` counts dashes from the nearest one you clicked (0, 1, 2, …); `near`/`far`
is the end of that dash. Measure the geometry on site; markings vary.

With clicks on parallel lines only, the along-road offset between the lines
(`phase_m`) cannot be recovered from the image: every value fits equally well and
corresponds to a **shear** of the ground plane. A shear does not change distances
measured along a lane, so along-lane speeds are unaffected (this is tested in
`tests/test_calibration.py`); lateral positions during lane changes are affected.
Tape one transverse offset, or add a point on a transverse marking, if that
matters.

## Checking a calibration

`roadspeed calibrate` prints the reprojection residuals (pixels and metres) and
whether the zone corners fall inside the image, and recovers a pinhole camera
(focal length, height, tilt) from the homography as a plausibility check: the
recovered height should match the mounting height you measured.
`--check-image` draws the clicked points (yellow), the surveyed points re-projected
through the calibration (magenta), the lane boundaries and the zone limits.

## Uncertainty settings

`click_sigma_px` (per axis) and `ground_sigma_m` (per axis) feed the Monte-Carlo
calibration uncertainty. With six or more points the tool also estimates the click
noise from the residuals and uses the larger of the two. Set `ground_sigma_m` from
the repeated tape measurements.

## Optional: reference point above the road

The default reference point is the bottom edge of the vehicle's blob, which lies
(approximately) on the road. If you set `"speed": {"contact": "center",
"contact_height_m": h}`, the box centre is used and corrected for its height `h`
with the camera recovered from the homography. Simulation shows this is worse than
the bottom edge unless the height is known for each vehicle
(`results/e12_contact.json`).
