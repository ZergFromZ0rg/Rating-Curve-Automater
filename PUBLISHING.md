# Publishing a release

The README's install instructions assume `rating-curve-automater` is on PyPI.
The name is available and the package builds and passes `twine check`.

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

### One-time PyPI setup

1. Create a [PyPI](https://pypi.org/) account.
2. Add a *pending* trusted publisher at
   <https://pypi.org/manage/account/publishing/>:
   - PyPI Project Name: `rating-curve-automater`
   - Owner: `ZergFromZ0rg`
   - Repository name: `Rating-Curve-Automater`
   - Workflow name: `publish.yml`
   - Environment name: `pypi`
3. In the GitHub repo, create an Environment named `pypi`
   (Settings → Environments → New environment). No secrets needed; optionally
   add yourself as a required reviewer so each publish needs a click.

### Each release

1. Bump `pyproject.toml` version if needed; commit and push.
2. GitHub → Releases → **Draft a new release**.
   - Tag: `v0.1.0` (create on publish), target `main`.
   - Generate release notes.
   - **Publish release.**
3. The workflow runs: test → build → `twine check` → publish. Watch it under
   the Actions tab.
4. Verify from a clean environment:
   `pip install "rating-curve-automater[app]"` then `rca app`.

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
