# MakeHuman 1.x on macOS / Apple Silicon

This fork runs the MakeHuman 1.x desktop application natively on Apple Silicon Macs, from source,
with current Python, numpy 2 and PyQt5 5.15. Nothing runs under Rosetta.

## Quick start

```bash
macos/setup.sh      # once: .venv, dependencies, system assets, compiled data (idempotent)
macos/makehuman     # start the app (arguments are passed through, e.g. a .mhm file)
```

`setup.sh` uses [uv](https://docs.astral.sh/uv/) when it is installed and falls back to
`python3.12 -m venv` otherwise. Pick another Python with `PYTHON=3.13 macos/setup.sh`.
It refuses an x86_64 interpreter on Apple Silicon, because that would run under Rosetta.
Running it again does not download the assets again (`download_system_assets.py --force` does).

User files (models, exports, settings, logs) go to `~/Documents/MakeHuman/v1py3`, as upstream.
Set `MH_HOME_LOCATION` to use another folder.

## Tested configuration

| | |
|---|---|
| Machine | Apple M3 Max, macOS 26.6.2 |
| Python (arm64) | 3.11.13, 3.12.11, 3.13.8, 3.14.8: unit and end-to-end tests pass on all four |
| numpy | 2.4.6 / 2.5.4 (unit tests also pass on 1.26.4) |
| PyQt5 / Qt | 5.15.11 / 5.15.x arm64 wheels from PyPI |
| PyOpenGL | 3.1.10 |
| OpenGL | "2.1 Metal": Apple's OpenGL implemented on top of Metal, GLSL 1.20 |

## What was broken, and what changed

| Problem | Symptom | Fix |
|---|---|---|
| numpy 2 removed `ndarray.tostring()` and binary `np.fromstring()` | crash on start-up while loading the first texture; compiled meshes and proxies could not be loaded | `tobytes()` / `frombuffer()`; strings are encoded to UTF-8 explicitly, as `fromstring` did. Offsets are now byte offsets, which also fixes non-ASCII names (`core/files3d.py`, `shared/proxy.py`, `makehuman.py`, `lib/image_qt.py`, `lib/image.py`) |
| numpy 2: `np.array(x, copy=False)` raises when a copy is needed | posing and skeleton maths failed with list input | `np.asarray()` (`core/transformations.py`) |
| numpy 1.24 to 1.26 removed `np.bool` | `unique()` failed on empty arrays | builtin `bool` (`lib/unique.py`) |
| Invalid escape sequences | `SyntaxWarning` on Python 3.12+, a `SyntaxError` in a future Python | raw strings (10 places) |
| Source runs on macOS were treated as py2app bundles | the app only started from inside `makehuman/`; stdout/stderr went to block-buffered files instead of the terminal | bundle logic only when frozen; redirected logs are line buffered (`makehuman.py`) |
| Retina (HiDPI) displays: the GL framebuffer has 2x the logical pixels | mouse picking missed the body, camera depth queries landed ~87 units off, screen grabs and save thumbnails captured the wrong quarter of the view | picking, depth and grabs work in framebuffer pixels (`lib/glmodule.py`, `apps/gui/guisave.py`) |
| Quitting called `sys.exit()` inside a Qt slot | intermittent SIGSEGV on quit (the "Python quit unexpectedly" dialog) | leave the event loop with `quit()`, hide the main window before PyQt's exit cleanup (`lib/qtui.py`) |
| Socket server plugin dropped a running `QThread` on unload | SIGABRT on quit when the server was enabled | wait for the thread (`plugins/8_server_socket`) |
| `download_assets_git.py` clones `makehumancommunity/makehuman-assets`, which no longer exists | no hair, eyebrows, eyelashes, clothes, skins, proxies | `download_system_assets.py` installs the CC0 `makehuman_system_assets` pack from the MakeHuman Community static site (SHA-256 pinned, safe extraction, never overwrites) |
| That pack's materials use litsphere textures that are not in the pack or in this repository | missing textures logged as errors, wrong shading of clothes and hair | the 16 `lit_*.png` and `fedora_displacement.png` from the official MakeHuman 1.2.0 release (CC0), added under `makehuman/data` |
| Expressions, the `game_engine`, `cmu_mb` and `default_no_toes` skeletons and 15 poses also came from the vanished asset repository and are not in the pack | empty Expression tab, only the default skeleton, two poses | the 32 expressions, 3 skeletons and 15 poses of the official 1.2.0 release (CC0), added under `makehuman/data` |
| A material with a missing texture file aborted the Ogre export | `FileNotFoundError` | the texture is skipped with a warning (`shared/material.py`) |

The litspheres, expressions, skeletons and poses were taken from `makehuman-community-1.2.0-macos.zip`
on download.tuxfamily.org/makehuman/releases. Every file this repository already shared with that
archive is byte-identical to it, except `rigs/default_weights.mhw`, which is newer here and was kept.

## Tests

```bash
uv pip install --python .venv/bin/python -r requirements-dev.txt
macos/check.sh          # unit tests + coverage gate; no window opens
macos/check.sh --e2e    # also the end-to-end test, which opens the MakeHuman window
```

- **Unit tests** (`tests/unit`) cover every change above. Each one was checked to fail against
  the upstream code. The gate requires at least 90% line coverage of the code this fork adds
  (`download_system_assets.py`, currently 98%).
- **End-to-end test** (`tests/e2e/run_smoke.py`) starts the real app in a temporary home
  (`MH_HOME_LOCATION`), with an extra plugin that drives it through its own APIs: it opens all 51
  tasks, changes modifiers with undo/redo, checks picking and depth on the Retina framebuffer,
  grabs the screen, equips eyes, hair, eyebrows, eyelashes, teeth, tongue, clothes and a topology
  proxy, changes the skin, loads a skeleton and a pose, applies an expression, toggles smoothing and
  wireframe, cycles the cameras, randomizes, renders with and without anti-aliasing, saves and
  reloads an MHM, switches to the `game_engine` skeleton, runs every exporter, and queries the
  socket server. Errors that MakeHuman only logs count as failures. A
  watchdog outside the app kills it after `--timeout` seconds. `--coverage` reports which
  application lines the run executed: 62% of the app at the moment.

## Known limitations

- MakeHuman 1.x renders with legacy OpenGL 2.1. Apple deprecated OpenGL in macOS 10.14; it still
  works on macOS 26 through Metal, but the 1.x line would stop working if Apple removed it.
  MakeHuman 2 ([makehumancommunity/makehuman2](https://github.com/makehumancommunity/makehuman2))
  is the long-term path.
- This is a source install, not a signed `.app` bundle. The py2app recipe in
  `buildscripts/darwin` has not been updated.
- The in-app asset downloader (Community tab) cannot work: the asset database it downloads
  (`makehumancommunity.org/sites/default/files/assets.json`) returns 404 since the community site
  retired its legacy infrastructure. Asset packs are still published on
  [static.makehumancommunity.org](https://static.makehumancommunity.org/assets/assetpacks/index.html).
