# Publishing a release

The install instructions in the README assume `rating-curve-automater` is on
PyPI. Here is how to get it there and how to cut a release.

## One-time setup

- [ ] **Fix the copyright holder.** `LICENSE` and `pyproject.toml`
      (`authors = [{ name = "ZergFromZ0rg" }]`) currently use a placeholder.
      Replace with the real name/entity before the first upload — it is baked
      into every published wheel.
- [ ] Create a [PyPI](https://pypi.org/) account and a project-scoped API token
      (or configure [Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
      from GitHub Actions, which needs no token).
- [ ] Decide the release owner. Once `0.1.0` is uploaded to PyPI that version
      number is permanently taken — you cannot re-upload different code as
      `0.1.0`, only yank it.

## Versioning

The version is single-sourced from `pyproject.toml` (`project.version`);
`rating_curve_automater.__version__` reads it back from the installed
distribution metadata. To bump: edit `pyproject.toml`, commit, tag.

## Cut a release

```bash
# 1. Bump pyproject.toml version, commit.
git commit -am "release: v0.1.0"

# 2. Tag it (annotated).
git tag -a v0.1.0 -m "v0.1.0"
git push origin main --tags

# 3. Build the sdist + wheel into dist/.
python3 -m pip install --upgrade build twine
python3 -m build

# 4. Sanity-check the artifacts.
python3 -m twine check dist/*
#    optionally: pip install dist/rating_curve_automater-0.1.0-py3-none-any.whl
#    in a clean venv and run `rca app`.

# 5. Upload.
python3 -m twine upload dist/*
```

`uv` users can replace steps 3 and 5 with `uv build` and `uv publish`.

## After upload

- [ ] Create a GitHub Release from the tag, pasting the changelog since the last
      tag (`git log --oneline PREV_TAG..v0.1.0`).
- [ ] Verify `pip install "rating-curve-automater[app]"` works from a fresh venv
      on a machine that has never seen the repo.
