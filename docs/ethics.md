# Ethics, privacy and legal position

## What this tool is

An **audit instrument for road-safety planning**. It estimates how fast traffic
moves on one road segment so that a municipality and the public can see whether
speeds are compatible with the street's use (for example where people walk), and
can judge whether a change — traffic calming, a crossing, signage, a lower limit —
worked. Its outputs are aggregate statistics: mean speed, 85th-percentile speed,
share of vehicles over the limit, flow, by lane and direction, each with a
confidence interval.

## What it is not

- **Not a ticketing or enforcement device.** It must not be used to fine, warn,
  identify or prosecute anyone. Its per-vehicle speeds carry an uncertainty and a
  quality flag; they are inputs to statistics, not evidence about a driver.
- **Not certified.** It has no type approval as a measuring instrument and has not
  been validated in the field yet. Everything measured so far was measured on
  simulated video (see the README). Before statistics from a real site are
  published, the camera at that site must pass a validation like
  [field-protocol.md](field-protocol.md).
- **Not a surveillance system.** It contains no number-plate reading, no face
  detection and no re-identification of vehicles across sites, and such features
  will not be added (see CONTRIBUTING.md).

## Data minimisation by design

- **Offline first.** Processing runs on a local computer; nothing is uploaded and
  no model is downloaded by default.
- **Numbers only.** The outputs are records with time, lane, direction, speed,
  uncertainty and quality flags (`records.csv` / `records.json`). No images are
  written by default.
- **Review crops are optional and anonymised.** With `--review-sheets`, flagged
  records get a sheet with crops that are pixelated to a coarse grid and blurred
  before being written, so plates and faces are not legible, plus a
  position-versus-time plot. Full frames are never written by the pipeline.
- **Raw video** is the most sensitive item. In a pilot it stays on an encrypted,
  offline disk, is accessible to named people only, and is deleted once validation
  and review are complete (at most 30 days unless a shorter period is agreed).
  See the retention section of the field protocol.

## Human review

Records that fail quality checks (merged blobs, occlusion gaps, ambiguous lane,
large residuals, high uncertainty) are not silently used or silently dropped: they
are listed in `review_queue.json` with reasons. They are excluded from statistics
until a reviewer accepts them (`roadspeed review`), and every decision is stored
in `review_decisions.json`. The report states how many records were valid,
flagged, accepted, rejected and pending, because excluding difficult cases can
bias the statistics (for example if heavy vehicles are flagged more often).

## Honest reporting

- Every statistic is published with its confidence interval, the number of
  vehicles behind it, the observation period and the data-quality counts.
- The uncertainty model and its known blind spots are documented (calibration and
  pixel noise are propagated; systematic effects such as the reference point
  drifting along the vehicle body are not, and are measured in simulation and in
  the field validation instead).
- Simulated results are labelled as simulated, everywhere.

## Legal

- Lebanon: processing video of a public road may involve personal data. Check
  obligations under Law No. 81 of 2018 on Electronic Transactions and Personal Data
  ([UN ESCWA legislation database](https://alp.unescwa.org/legislations/law-81-2018-electronic-transactions-and-personal-data))
  with legal counsel, and operate under a written agreement with the
  municipality. This document is not legal advice.
- According to WHO's 2023 country profile for Lebanon, the national maximum urban
  speed limit is 50 km/h and local authorities can modify limits
  ([WHO, 2023](https://www.who.int/publications/m/item/road-safety-lbn-2023-country-profile)),
  which is why speed evidence at street level matters to a municipality.

## Public reporting

Publish, for each site: location and purpose of the study, the period, the
aggregated statistics with intervals, data-quality counts, the validation result,
and the method (this repository). Do not publish imagery of people or vehicles
that could identify them. Invite comment from residents, and report back what the
municipality decided to do with the results.
