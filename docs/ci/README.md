# CI workflow templates

These are the release-grade GitHub Actions workflows for Fermi. They are kept
here (instead of `.github/workflows/`) only because the token used to sync this
repository lacks the GitHub `workflow` scope, which GitHub requires for any
change to `.github/workflows/*`.

## Activate

Install them once, either:

- **Web UI** — create `.github/workflows/windows-build.yml` and
  `.github/workflows/macos-dmg.yml` and paste the contents of the files here; or
- **CLI** (with a token that has the `workflow` scope):

  ```bash
  gh auth refresh -h github.com -s workflow
  mkdir -p .github/workflows
  cp docs/ci/windows-build.yml .github/workflows/
  cp docs/ci/macos-dmg.yml     .github/workflows/
  git add .github/workflows && git commit -m "ci: add native build workflows"
  git push
  ```

## What they do

| File | Runner | Output |
|------|--------|--------|
| `windows-build.yml` | `windows-latest` | `Fermi.exe` (one-file, no console) artifact `Fermi-windows` |
| `macos-dmg.yml`     | `macos-latest`   | `Fermi.dmg` artifact `Fermi-macos-dmg` |

Both bundle `xraylib` via `pyinstaller --collect-all xraylib` — required because
`xraylib` ships compiled data tables that are otherwise missing from the frozen
app.

> The existing `.github/workflows/main.yml` already builds the macOS DMG; the
> template here additionally collects `xraylib` and adds the Windows build.
