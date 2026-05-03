Lightly modified fork of **BAMM: Bidirectional Autoregressive Motion
Model** (ECCV 2024)
. All credit for the model, training
recipe, and pretrained checkpoints goes to the original authors.

- Paper: [arXiv 2403.19435](https://arxiv.org/abs/2403.19435)
- Project page: <https://exitudio.github.io/BAMM-page/>
- Original code: <https://github.com/exitudio/BAMM>

```bibtex
@inproceedings{pinyoanuntapong2024bamm,
  title     = {BAMM: Bidirectional Autoregressive Motion Model},
  author    = {Pinyoanuntapong, Ekkasit and Saleem, Muhammad Usama and Wang, Pu
               and Lee, Minwoo and Das, Srijan and Chen, Chen},
  booktitle = {Computer Vision -- ECCV 2024},
  year      = {2024}
}
```

For full setup, training, evaluation, and the original CLI commands
(`gen_t2m.py`, `train_*.py`, `eval_*.py`), see the upstream README at the
project page above.

## What changed in this fork

- Added Flask + Three.js GUI mirroring the FloodDiffusion live-viewer (port `7860`).
- Added macOS support via a CUDA → MPS → CPU device fallback.
- Wired the paper's Joint2BVHConvertor IK refinement (`iterations=100`, `foot_ik=True`) into the live worker so the streamed joints match the paper's `_ik.mp4` output.
- Worker keeps regenerating new batches while the prompt is unchanged so playback never stops.
- `motion_length` exposed as a slider in the UI's Config modal (same range as `--motion_length`).
- Backend-only performance telemetry: `--profile` flag + `POST /api/export_performance` writes `perf_BAMM_<ts>.json` for offline comparison with MMM.
- See the project root's `COMPARISON.md` for the full paper-vs-fork diff.

## How to run

One-time setup is described in the upstream README (conda env, pretrained
checkpoint downloads). After that, from this directory:

```bash
python app.py
# open http://127.0.0.1:7860
```

Optional flags:

```bash
python app.py --port 7860 --profile --perf-dir ./
```

To dump a performance snapshot during a session:

```bash
curl -X POST http://127.0.0.1:7860/api/export_performance
```

## License

Same as upstream — see `LICENSE-CC-BY-NC-ND-4.0.md` (in the upstream repo)
and the licenses of all transitive dependencies (CLIP, SMPL-X, MoMask,
T2M-GPT, HumanML3D).
