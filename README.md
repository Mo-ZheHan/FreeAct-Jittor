<h1 align="center">FreeAct: Demonstration-Free Robot Adaptation<br>via Action-Grounded Generated Videos</h1>

<p align="center"><strong>NeurIPS 2026</strong> · Jittor implementation</p>

<p align="center">
  Zhe-Han Mo · Jia-Ning Li · Jiaxi Song · Meng-Hao Guo · Shi-Min Hu<sup>*</sup><br>
  BNRist, Department of Computer Science and Technology, Tsinghua University<br>
  <sub><sup>*</sup>Corresponding author</sub>
</p>

<p align="center">
  <a href="https://freeact.pages.dev/FreeAct.pdf"><b>Paper PDF</b></a> &nbsp;·&nbsp;
  <a href="https://freeact.pages.dev/"><b>Project Page</b></a> &nbsp;·&nbsp;
  <a href="https://github.com/Mo-ZheHan/FreeAct"><b>PyTorch Code</b></a> &nbsp;·&nbsp;
  <a href="https://huggingface.co/Mo-ZheHan/FreeAct"><b>Models</b></a> &nbsp;·&nbsp;
  <a href="https://huggingface.co/datasets/Mo-ZheHan/FreeAct-Data"><b>Data</b></a>
</p>

## Overview

FreeAct adapts a pretrained vision-language-action (VLA) policy to a new environment without target-domain expert demonstrations: a physically grounded latent action model pseudo-labels generated videos of the target tasks, and the policy is trained on them together with source-domain demonstrations.

This repository implements FreeAct inference in [Jittor](https://github.com/Jittor/jittor). It loads the released PyTorch checkpoints directly and covers policy inference, closed-loop evaluation on the target tasks and the two additional target environments, and the latent action analysis (RQ1) with the FreeAct, UniVLA and LAPA tokenizers. Training is provided in the [PyTorch implementation](https://github.com/Mo-ZheHan/FreeAct).

## Installation

Tested on Linux with Python 3.11 and NVIDIA RTX 4090 GPUs. Dependencies are pinned in `uv.lock` and installed with [uv](https://docs.astral.sh/uv/getting-started/installation/); Jittor also needs a C++ compiler.

```bash
git clone https://github.com/Mo-ZheHan/FreeAct-Jittor.git
cd FreeAct-Jittor
uv sync
source .venv/bin/activate
cp env.sh.example env.sh
source env.sh
```

`env.sh` points Jittor to the environment's `python3.11-config`; activate the environment and source it in every new shell or batch job. The first run compiles Jittor's operators, which takes about 10 minutes.

## Quick start

```bash
hf download Mo-ZheHan/FreeAct --include 'stage3/*' --local-dir checkpoints/FreeAct
bash scripts/eval.sh checkpoints/FreeAct/stage3
```

This runs 50 rollouts per task on Tasks A–E on all visible GPUs (about 3 hours on two RTX 4090s) and writes the success rates to `outputs/eval/<run>/results.json`. Append `--n_episodes 10` for a quick check.

## Evaluation

| Suite | Tasks | Description |
|---|---|---|
| `libero_target` (default) | A–E (IDs 0–4) | Main target environment |
| `libero_camera` | B (ID 0) | Camera azimuth rotated by +5° |
| `libero_bottle` | C (ID 0) | Milk carton replaced by a ketchup bottle |

Each task uses 50 fixed initial states and seed 42; the policy samples tokens at temperature 0.75 and replans after every action.

```bash
bash scripts/eval.sh <checkpoint> [suite] [options]
bash scripts/eval.sh checkpoints/FreeAct/self_evolve
bash scripts/eval.sh checkpoints/FreeAct/stage3 libero_camera
```

Any policy directory in the [model zoo](https://github.com/Mo-ZheHan/FreeAct#model-zoo) (`stage3`, `self_evolve*`, `baselines/*/stage3`) is downloaded with `--include '<dir>/*'` and evaluated the same way. Useful options are `--n_episodes`, `--task_ids`, `--record_video` and `--save_rollouts`; see `python -m freeact.eval.run --help`.

Per-task success rates (%):

| Policy | A | B | C | D | E | Average |
|---|---:|---:|---:|---:|---:|---:|
| Without adaptation | 0 | 8 | 0 | 0 | 0 | 1.6 |
| `stage3` | 36 | 78 | 80 | 82 | 8 | 56.8 |
| `self_evolve` | 76 | 84 | 100 | 56 | 52 | 73.6 |

In the two additional target environments, `stage3` reaches 74 (camera shift, Task B) and 60 (ketchup bottle, Task C), and self-evolution raises them to 92 and 94.

## Latent action analysis (RQ1)

The analysis needs the calibrated DROID splits (about 100 GB) and the latent action model:

```bash
hf download Mo-ZheHan/droid --repo-type dataset --include '*_cam/*' --local-dir data/droid
hf download Mo-ZheHan/FreeAct --include 'lam/*' --local-dir checkpoints/FreeAct
python -m freeact.analysis.extract --lam checkpoints/FreeAct/lam/lam.ckpt --name freeact
python -m freeact.analysis.linear_probe --names freeact
python -m freeact.analysis.nldr --names freeact
python -m freeact.analysis.umap_plot --names freeact
```

Extraction samples 50k training and 20k held-out frame pairs; results are saved to `outputs/analysis`. To compare with UniVLA and LAPA, extract their tokenizers too and pass all names:

```bash
hf download qwbu/univla-latent-action-model lam-stage-2.ckpt --local-dir checkpoints/univla
hf download latent-action-pretraining/LAPA-7B-openx laq_openx.pt --local-dir checkpoints/lapa
python -m freeact.analysis.extract --lam checkpoints/univla/lam-stage-2.ckpt --name univla
python -m freeact.analysis.extract --lam checkpoints/lapa/laq_openx.pt --name lapa
python -m freeact.analysis.linear_probe --names freeact univla lapa
```

## Testing

```bash
python -m unittest discover tests   # checks the target suites; no GPU needed
```

## Repository structure

```text
FreeAct-Jittor/
├── scripts/            # evaluation launcher
├── freeact/
│   ├── vla/            # Qwen3-VL policy, LoRA, processor and continuous action head
│   ├── lam/            # latent action model with the DINOv3 backbone
│   ├── tokenizers/     # FreeAct, UniVLA and LAPA latent action tokenizers
│   ├── envs/           # target task suites, scenes and initial states
│   ├── eval/           # distributed closed-loop evaluation (Ray)
│   └── analysis/       # RQ1: feature extraction, linear probing, NLDR, UMAP
└── tests/
```

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{mo2026freeact,
  title     = {FreeAct: Demonstration-Free Robot Adaptation via Action-Grounded Generated Videos},
  author    = {Mo, Zhe-Han and Li, Jia-Ning and Song, Jiaxi and Guo, Meng-Hao and Hu, Shi-Min},
  booktitle = {The Fortieth Annual Conference on Neural Information Processing Systems},
  year      = {2026},
  url       = {https://openreview.net/forum?id=wNWxVVntG2}
}
```

## Acknowledgements

This implementation is built on [Jittor](https://github.com/Jittor/jittor) and follows [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL), [UniVLA](https://github.com/OpenDriveLab/UniVLA), [LAPA](https://github.com/LatentActionPretraining/LAPA), [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) and [DINOv3](https://github.com/facebookresearch/dinov3). It ports code from [Transformers](https://github.com/huggingface/transformers), [PEFT](https://github.com/huggingface/peft), [timm](https://github.com/huggingface/pytorch-image-models), [DINOv2](https://github.com/facebookresearch/dinov2) and [LeRobot](https://github.com/huggingface/lerobot). We thank the authors of [DROID](https://droid-dataset.github.io/) for releasing their dataset, and [OpenVLA](https://github.com/openvla/openvla) and [IPEC-COMMUNITY](https://huggingface.co/IPEC-COMMUNITY) for the processed LIBERO demonstrations.

## License

This repository is released under the [Apache-2.0 License](LICENSE); the LAPA-derived code in `freeact/tokenizers/laq` keeps its [MIT License](freeact/tokenizers/laq/LICENSE). Ported code and third-party models and data are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md); in particular, `lam.ckpt` contains DINOv3 weights under the [DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md). Checkpoints and data are released under the licenses on their Hugging Face cards.
