# Publishing a release

`rating-curve-automater` is published at
<https://pypi.org/project/rating-curve-automater/> (first release: `0.1.0`,
2026-09-02; latest: `0.3.0`, 2026-09-03). This is the guide for cutting the next
one.

## Versioning

The version is single-sourced from `pyproject.toml` (`project.version`);
`rating_curve_automater.__version__` reads it back from the installed
distribution metadata. To bump: edit `pyproject.toml`, commit, then make a
GitHub Release whose tag matches (`v0.1.0`).

Once a version is uploaded to PyPI that number is permanently taken — you can
only yank it, never re-upload different code as the same version.

---

## Recommended: automated publish via GitHub Releases (no tokens)

`.github/workflows/publish.yml` builds and publishes whenever you publish a
GitHub Release, using PyPI **Trusted Publishing** (OIDC). No API token or secret
is stored.

### One-time setup (done for 0.1.0 — kept here for reference)

1. PyPI account.
2. Trusted publisher at <https://pypi.org/manage/account/publishing/>:
   project `rating-curve-automater`, owner `ZergFromZ0rg`, repo
   `Rating-Curve-Automater`, workflow `publish.yml`, environment `pypi`.
3. GitHub repo Environment named `pypi` (Settings → Environments). No secrets;
   optionally require a reviewer so each publish needs a click.

### Each release

**1. Prep the working tree (commit + push, wait for CI green).**

- Bump `project.version` in `pyproject.toml`.
- Changelog lives in **`README.md` under `## Changelog`** (there is no
  `CHANGELOG.md`). Rename the **`**Unreleased**`** block to `**vX.Y.Z**`
  `(current release)` with today's date, and drop `(current release)` from the
  previous version.
- If any dependency floor moved, say so in the changelog and check
  `requirements.txt` / `environment.yml` don't pin it lower.
- Run the **full** suite locally *with the `[bayesian]` extra* — CI skips those
  ~12 slow tests (`pip install -e ".[app,dev,bayesian]"; pytest -q`).

**2. Publish the GitHub Release.**

- GitHub → Releases → **Draft a new release**.
- Tag `vX.Y.Z` (create on publish), target `main`; generate release notes (or
  paste the `README.md` changelog block).
- **Publish.** If the `pypi` environment has a required reviewer, approve the
  run when it pauses.

**3. Watch `publish.yml`** (Actions tab): test → build → `twine check` →
`pypa/gh-action-pypi-publish` (Trusted Publishing, OIDC). All three jobs must be
green.

**4. Verify from a clean environment** (a *fresh* venv, and run it from **outside
the repo** or `import` picks up `./rating_curve_automater/`):

```bash
python3 -m venv /tmp/rca && /tmp/rca/bin/pip install "rating-curve-automater[app]"
cd /tmp
/tmp/rca/bin/rca --version                        # -> the new X.Y.Z
/tmp/rca/bin/rca fit --help                       # new flags present?
/tmp/rca/bin/python -c "import rating_curve_automater as r; print(r.__version__)"
# end-to-end: validate a workbook -> fit -> export_report, open the .xlsx
/tmp/rca/bin/rca app                              # boots headless, no traceback
```

`publish.yml` goes green a minute or two before PyPI serves the files —
`pypi.org/pypi/<pkg>/json` (`info.version`) and `pypi.org/simple/<pkg>/` lag on
the CDN. Give it ~5 min before deciding something is wrong.

**Known wart:** `rca app` runs `streamlit run` on the packaged `app.py` but does
**not** forward extra `streamlit` args (`--server.port`, …). Not a blocker; note
it if a user asks.

Tags so far: `v0.1.0`=`e150ec1`, `v0.2.0`=`1b3b4f3`, `v0.3.0`=`5546eb6`.

---

## Fallback: manual upload from your laptop

If you'd rather not use CI:

```bash
python3 -m pip install --upgrade build twine
python3 -m build                       # -> dist/*.whl, dist/*.tar.gz
python3 -m twine check dist/*
python3 -m twine upload dist/*          # prompts for credentials
```

At the prompt use `__token__` as the username and a PyPI API token
(`pypi-…`, created at <https://pypi.org/manage/account/token/>) as the password,
or put them in `~/.pypirc`. Then tag the release in git:

```bash
git tag -a v0.1.0 -m "v0.1.0" && git push origin main --tags
```
