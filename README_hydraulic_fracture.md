# XFEM Hydraulic Fracturing Demo

This patch implements a simplified 2D LEFM/XFEM hydraulic-fracture demo on top of the existing single-crack solver.

## Model used

For each crack and each global pressure step:

1. Solve the existing XFEM problem for that crack.
2. Compute mechanical `KI/KII` with `computeStress<13>`.
3. Add hydraulic opening contribution to Mode I:

```text
K_I,total = max(0, K_I,mechanical) + p * sqrt(pi * a_eff)
a_eff = max(total_crack_length / 2, element_size)
```

4. Grow each crack with existing `growCrackOneStep(...)` using physical `KIC` from `mesh/hydrofracture.txt`.

This is not a fully coupled fluid-flow / aperture / leakoff / pressure diffusion solver. It is a practical pressure-driven fracture-growth demo.

## Files

- `main.cpp` — replace your current `main.cpp`.
- `shaders/overlay_2d.vert`, `shaders/overlay_2d.frag` — legend color bar shaders.
- `shaders/overlay_text.vert`, `shaders/overlay_text.frag` — legend text shaders.
- `mesh/hydrofracture.txt` — hydraulic parameters.
- `mesh/cracks.txt` — multi-crack input file.

## Controls

- `Right` / `D` — next frame.
- `Left` / `A` — previous frame.
- `Space` — play / pause.
- `R` — reset to first frame.
- `Esc` — close.

## hydrofracture.txt format

```text
pressure_start pressure_step pressure_max KIC da_factor
```

Example:

```text
100000 150000 3000000 750000 0.50
```

Units:

- pressure values: Pa.
- `KIC`: Pa*sqrt(m).
- `da_factor`: dimensionless, `da = da_factor * min(wh, hh)`.

For glass start with:

```text
KIC = 0.75e6 Pa*sqrt(m)
E   = 70e9 Pa
nu  = 0.23
```

These are set in `main.cpp`.
