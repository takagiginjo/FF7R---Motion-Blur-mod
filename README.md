# Probe for Assassin's Creed Unity, Syndicate and III Remastered

A ReShade add-on that changes nothing in the game. It writes a log of what the
game does with TXAA and motion vectors, and saves the shaders that mention
motion, velocity, temporal, TXAA, TAA, reprojection, jitter or history next to
the add-on, in a `...-shaders` folder.

Copy `00-ACUnity-Probe.addon64` (or the Syndicate or AC3 Remastered one) next to the game's exe,
with ReShade installed (the version with add-on support), turn the anti-aliasing and the motion blur on in the
game's options, play for two minutes in the open, and send the `.log` file. If
a `...-shaders` folder appears, send the files in it as well.
