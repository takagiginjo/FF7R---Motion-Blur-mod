# Final Fantasy VII Remake — Camera Motion Blur

> **Status: Stable — tested in-game by the maintainer.** Use the tested build; development snapshots and experimental revisions may require separate validation.


A ReShade camera motion blur add-on for Final Fantasy VII Remake.

## Installation

Install ReShade with full add-on support for the game, then place `00-FF7Remake-CameraBlur.addon64` beside the game's executable. Keep the `00-` prefix for load ordering. Use the maintainer-tested build.

## Build from source

The CMake project uses C++17, the Windows SDK FXC shader compiler, and ReShade and ImGui headers under `deps/`. The output is `00-FF7Remake-CameraBlur.addon64`.
