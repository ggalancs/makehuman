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
| Modifier sliders applied changes only on mouse release | arrow keys moved the slider without changing the model (what was saved disagreed with what was shown); a value typed in the box could not be undone | keyboard, wheel and typed changes are applied once, with one undo step (`lib/modifierslider.py`, Measure tab) |
| Renders were read as BGR and Qt images built from RGBA as BGRA | colours were right on screen only because the two swaps cancelled; render data, RGBA images resized through Qt (SSS light map, AO map) had red and blue swapped | render data is RGB, `Image.toQImage` converts RGBA properly, the anti-aliasing workaround is gone (`lib/glmodule.py`, `lib/image.py`) |
| ASCII FBX properties were written as Python bytes | `P: "b'Path'", "b'KString'"…`: importers ignored texture paths, colours and bone sizes | text in the file (`plugins/9_export_fbx`) |
| A window saved in macOS full screen | the next start failed (`showFullscreen` does not exist) | `showFullScreen` (`lib/qtui.py`) |
| Several crashes found by the new tests | loading an MHM with a blank line before `version`; proxies with `author`/`license` lines or UV layers; non-square thumbnails; an empty or replaced scroll area (process abort); rotations of 120 degrees or more in BVH/pose maths; FBX/OBJ export of meshes without UVs | fixed, each with the test that failed before (see the commit history) |
| The Community tab said "Asset DB is now synchronized" when nothing could be downloaded | an empty asset list with a success message (the server answers 404) | the error is reported and the local list kept (`plugins/8_asset_downloader`) |

Unit and end-to-end tests written for this fork found many more defects in the inherited code.
The ones a user can reach are fixed, each with a test that failed before the fix; the rest, mostly
in code nothing calls, stay documented as strict `xfail` tests (35 cases). Code that could not
run was removed: the old asset downloaders, `shared/cache.py`, `shared/subtexture.py`,
`shared/mh2renderman.py`, the background plugin's texture projection, the disabled FBX animation
writer and Blender leftovers in the FBX exporter.

The litspheres, expressions, skeletons and poses were taken from `makehuman-community-1.2.0-macos.zip`
on download.tuxfamily.org/makehuman/releases. Every file this repository already shared with that
archive is byte-identical to it, except `rigs/default_weights.mhw`, which is newer here and was kept.

## Tests

```bash
uv pip install --python .venv/bin/python -r requirements-dev.txt
macos/check.sh          # unit tests + coverage gate of this fork's own code; no window opens
macos/check.sh --e2e    # also the end-to-end test and the 90% whole-app coverage gate
macos/coverage.sh       # just the whole-app coverage (report in local/coverage/report.txt)
```

- **Unit tests** (`tests/unit`, about 1900) cover the libraries, file formats, the human model,
  the Qt widgets and several plugins, headless (Qt `offscreen`). They need the system assets
  (`download_system_assets.py`) and compiled data, as CI installs them. The gate requires at
  least 90% line coverage of the code this fork adds (`download_system_assets.py`, 99%).
- **End-to-end test** (`tests/e2e/run_smoke.py`) starts the real app in a temporary home
  (`MH_HOME_LOCATION`), with an extra plugin that drives it through its own APIs: every task,
  modifiers with undo/redo (sliders driven by value box, drag and keyboard), picking, depth and
  screen grabs on the Retina framebuffer, every proxy type, skins, skeletons, poses, expressions,
  animation playback, renders (colours checked in the data, on screen and in the saved file),
  every exporter and its options, MHM save/load, the socket server, scripting, the shell, mass
  produce, scene editor, material editor, expression mixer, custom targets, garment foot poses,
  every Settings tab and installing user plugins. Errors that MakeHuman only logs count as
  failures; a watchdog outside the app kills it after `--timeout` seconds.
  The window opens on the main screen and the app runs as a background process, so a test run
  never takes the keyboard focus; dialogs are kept on the main window's screen.
- **Whole-app coverage**: unit + end-to-end tests execute **92%** of the application's lines
  (32,279 statements). `macos/coverage.sh` and `macos/check.sh --e2e` fail below 90%. CI runs the
  unit tests only (no display); the whole-app gate is run locally.

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
