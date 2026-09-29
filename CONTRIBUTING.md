# Contributing to Rating Curve Automater

Thanks for helping improve Rating Curve Automater. Contributions that improve
measurement validation, model transparency, documentation, accessibility, or
test coverage are especially welcome.

## Before you start

For substantial changes, open an issue first so the intended behavior and scope
are clear. Small documentation fixes and focused bug fixes can go straight to a
pull request.

Please do not include real monitoring data, credentials, or other sensitive
material in issues, examples, tests, or commits. Use synthetic or anonymized
measurements instead.

## Local development

Python 3.10–3.13 is supported. From a fresh clone:

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[app,dev]"
```

Run the test suite and basic checks before opening a pull request:

```bash
python -m pytest -q
python -m compileall -q rating_curve_automater
```

The optional Bayesian tests require the additional `bayesian` extra:

```bash
pip install -e ".[app,dev,bayesian]"
```

The bundled dataset and the seeded validation study provide reproducible ways
to exercise behavior beyond the unit tests. See `benchmarks/README.md` before
running the longer study.

## Coding conventions

- Keep public behavior documented in the README and tested in `tests/`.
- Prefer small, focused changes with clear error messages and deterministic
  tests.
- Preserve units and uncertainty information through the workflow; add a test
  when changing either.
- Keep dependencies in `pyproject.toml`; do not add an untracked local install
  recipe.
- Use type hints and the existing naming/style conventions in nearby code.
- Never commit secrets, private field data, generated reports, or scratch work.

## Commits and pull requests

Use short, imperative commit subjects with a conventional prefix where useful,
for example `fix(loader): handle date ranges` or `docs: clarify Bayesian setup`.
Keep unrelated changes in separate commits when practical.

Pull requests should explain:

1. What changed and why.
2. How the change was tested, including the Python version and optional extras.
3. Any user-facing behavior, compatibility, or scientific interpretation
   changes.
4. Whether documentation, changelog, or release notes need updating.

All pull requests must pass the GitHub Actions checks. Maintainers may ask for
an additional synthetic example or validation result for changes affecting
fitting, uncertainty, drift detection, or report output.

## Reporting security issues

Please follow [SECURITY.md](SECURITY.md) for suspected vulnerabilities. Do not
publish credentials or exploit details in a public issue.
