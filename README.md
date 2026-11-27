# Final Fantasy VII Remake — Camera Motion Blur

> **Status: Stable — tested in-game by the maintainer.** Use the tested build; development snapshots and experimental revisions may require separate validation.

## Downloads

- [Releases](https://github.com/takagiginjo/FF7R---Motion-Blur-mod/releases) — published packages, when available.
- [Successful GitHub Actions build](https://github.com/takagiginjo/FF7R---Motion-Blur-mod/actions/runs/46009018145) — artifact `motionblur`. Sign in to GitHub to download the ZIP, then extract it before installation. Workflow success confirms compilation, not independent gameplay validation of that exact binary.
- [ReShade 6.8.0](https://github.com/crosire/reshade/releases/tag/v6.8.0) — use the installer with full add-on support.


A ReShade camera motion blur add-on for Final Fantasy VII Remake.

## Installation

Install ReShade with full add-on support for the game, then place `00-FF7Remake-CameraBlur.addon64` beside the game's executable. Keep the `00-` prefix for load ordering. Use the maintainer-tested build.

## Configuration

Open the ReShade overlay (Home by default) and locate **Camera Blur**.

| Setting | Description |
| --- | --- |
| Enabled | Turns the effect on or off. |
| Use this add-on's blur | Selects the custom camera and object blur renderer. |
| Motion blur | Intensity, 0–100%. |
| Overall scale | Multiplier, 10–100%. |
| Longest blur | Low, Medium or High streak limit. |
| Keep near things sharp | Near-distance attenuation (0–10 m). |
| Show motion (debug) | Visualizes motion vectors; leave disabled for normal play. |

## Build from source

The CMake project uses C++17, the Windows SDK FXC shader compiler, and ReShade and ImGui headers under `deps/`. The output is `00-FF7Remake-CameraBlur.addon64`.

### Windows build

Run from the repository root with Visual Studio 2022 and the Windows SDK installed:

```powershell
git clone --depth 1 --branch v6.8.0 https://github.com/crosire/reshade.git deps/reshade
git clone https://github.com/ocornut/imgui.git deps/imgui
git -C deps/imgui checkout 3912b3d9a9c1b3f17431aebafd86d2f40ee6e59c
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

Alternatively run the Build workflow in GitHub Actions.

## Troubleshooting

Install only one version of this add-on at a time. If the effect does not appear, confirm ReShade was installed with full add-on support and check `ReShade.log`. Include the game version, GPU, add-on build and relevant logs when reporting issues.
