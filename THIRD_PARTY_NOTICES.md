# Third-party notices

FreeAct-Jittor is released under the [Apache-2.0 License](LICENSE). Parts of it are ported from other open-source projects, and it uses third-party models and data that are not covered by that license. The license terms of each original work continue to apply to the corresponding parts. Every ported file names its source in its header.

## Code ported to Jittor in this repository

| Source | License | Files |
|---|---|---|
| [Transformers](https://github.com/huggingface/transformers) Qwen3-VL model and processor | Apache-2.0 | `freeact/vla/qwen3_vl.py`, `freeact/vla/processor.py` |
| [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) finetuning code | Apache-2.0 | `freeact/vla/data/rope2d.py` |
| [PEFT](https://github.com/huggingface/peft) LoRA layers | Apache-2.0 | `freeact/vla/lora.py` |
| [timm](https://github.com/huggingface/pytorch-image-models) `Eva` (DINOv3 forward) | Apache-2.0 | `freeact/lam/modules/dinov3.py` |
| [DINOv2](https://github.com/facebookresearch/dinov2) | Apache-2.0 | `freeact/tokenizers/dinov2.py` |
| [UniVLA](https://github.com/OpenDriveLab/UniVLA) | Apache-2.0 | `freeact/lam/modules/blocks.py`, `freeact/tokenizers/univla.py`, `freeact/vla/action_decoder.py` |
| [LAPA](https://github.com/LatentActionPretraining/LAPA), building on [phenaki-pytorch](https://github.com/lucidrains/phenaki-pytorch) and [NSVQ](https://github.com/MHVali/Noise-Substitution-in-Vector-Quantization) | MIT | `freeact/tokenizers/laq/` (license and copyright notices in [`freeact/tokenizers/laq/LICENSE`](freeact/tokenizers/laq/LICENSE)) |
| [LeRobot](https://github.com/huggingface/lerobot) dataset reader and LIBERO environment wrapper | Apache-2.0 | `freeact/lerobot.py`, `freeact/envs/libero_env.py` |

Copyright notices retained from the ported code:

- Transformers Qwen3-VL: Copyright 2025 The Qwen Team and The HuggingFace Inc. team; Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team (Apache-2.0).
- DINOv2: Copyright (c) Meta Platforms, Inc. and affiliates (Apache-2.0).
- LAPA: Copyright (c) 2024 Seonghyeon Ye and Joel Jang (MIT).
- phenaki-pytorch: Copyright (c) 2022 Phil Wang (MIT).
- NSVQ: Copyright (c) 2022-present Mohammad Hassan Vali (MIT).

## Models

- **DINOv3 ViT-B/16** (`vit_base_patch16_dinov3.lvd1689m`) is the frozen backbone of the latent action model, and its weights are contained in the released `lam/lam.ckpt`. They are released by Meta under the [DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md), whose terms apply to their use and redistribution.
- **Qwen3-VL-4B-Instruct** ([Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct), Apache-2.0) is the base model of the policy and is downloaded at runtime.
- **Baseline tokenizers**, downloaded by the user: UniVLA [`lam-stage-2.ckpt`](https://huggingface.co/qwbu/univla-latent-action-model) (Apache-2.0), whose backbone is DINOv2 (Apache-2.0), and LAPA [`laq_openx.pt`](https://huggingface.co/latent-action-pretraining/LAPA-7B-openx) (MIT).

## Simulation, libraries and data

- [Jittor](https://github.com/Jittor/jittor) (Apache-2.0) is the deep learning framework.
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) (MIT) and [robosuite](https://github.com/ARISE-Initiative/robosuite) (MIT) provide the simulated environments. `freeact/envs` patches LIBERO's `libero_spatial` scenes at runtime and adds its own initial states and texture.
- [DROID](https://droid-dataset.github.io/) (CC BY 4.0) is used for the latent action analysis; the released conversion and its license are described on its [data card](https://huggingface.co/datasets/Mo-ZheHan/droid).
