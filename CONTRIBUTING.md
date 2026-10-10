# Contributing

Thanks for your interest. This is a small research tool; contributions that make
its measurements more trustworthy are the most welcome.

- **Set up:** `make setup` (creates `.venv`), then `make test` and `make lint`.
- **Before a pull request:** tests pass, `ruff check` and `ruff format --check` are
  clean, and new behaviour has a test. If you change anything that affects speed
  estimates, run `make reproduce-quick` and say in the pull request how the quick
  results moved.
- **Numbers in documentation** must come from a file the code wrote (`results/`).
  Do not hand-edit results or figures; regenerate them.
- **Privacy:** never commit video, frames or crops of real people or number plates.
  Test data must be synthetic or explicitly licensed and anonymised.
- **Scope:** this is an audit tool for road-safety planning, not an enforcement
  device. Features whose main use is identifying individuals (plate reading, face
  capture) will not be accepted.
