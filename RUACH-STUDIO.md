# Ruach Studio's ggml

This branch, `ruach-studio`, is [Ruach Studio](https://github.com/igrbible/Ruach_Studio)'s derivative of [ggml](https://github.com/ggml-org/ggml): the exact ggml that the Studio's engine is built on, with our changes on top as commits of their own. We hide nothing and are not ashamed of it: what we run is here, change by change, for anyone to read, check and take.

## The base

`21013893` «cpu: add fused snake activation» from [ServeurpersoCom/ggml](https://github.com/ServeurpersoCom/ggml), the ggml that [yue2.cpp](https://github.com/ServeurpersoCom/yue2.cpp) builds on: ggml-org's `57e2c2ca` (ggml 0.29.0) with that one commit. Ruach Studio moved to it in HERESY 2240.

## What we change

| commit | HERESY | what it does |
|---|---|---|
| cuda: flash attention mask scan reads a mask shared by every sequence in place | 1169, 1212 (moved in 2240) | `src/ggml-cuda/fattn-common.cuh`: when several sequences share one mask (`mask->ne[3] == 1`), the scan for each tile's last key read past the mask's end from the second sequence on (an illegal memory access, or wrong limits where it did not fault). A one-line fix that stands on its own. |
| cuda: an F32 precision request on src1 is met by an F32 GEMM without TF32 | 2329 | `src/ggml-cuda/ggml-cuda.cu`: a matrix product whose src1 asks for F32 (`ggml_prec_set_src(..., GGML_PREC_F32, 1)`) skips the mma kernels and runs cuBLAS with TF32 off for that call only, so a float32 decoder decodes in float32 on CUDA. Products that ask for nothing are unchanged. |

The HERESY numbers are entries of our patch log, `heresy/docs/HERESY.md` in the Ruach Studio repository, where each change is told with how it was found and how it was proven. In the Studio the same changes are the patch files `build/ggml-patches/0001-*.patch` and `0002-*.patch`, put on this ggml before the engine is built.

The tag `ruach-old-pin-cc5df49a` keeps the ggml the Studio ran on before HERESY 2240: ServeurpersoCom's `cc5df49a` (ggml 0.27.0 with his fused snake and a SYCL get_rows fix), which his force-pushed master no longer holds.

## Licence

ggml is under the MIT licence (`LICENSE`), and every file of it keeps its notice. As Ruach Studio's release states it, what we wrote, the patches over the engine among it, is under the GNU Affero General Public License, version 3 or any later version (AGPL-3.0-or-later), with the additional terms of its section 7 (`LICENSE` and `LICENSE-ADDITIONAL-TERMS.md` in the Ruach Studio repository).

`master` of this fork is ggml-org's, untouched.
