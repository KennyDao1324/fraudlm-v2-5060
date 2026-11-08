# FraudLM v2

GPT-2 XL with rank-16 LoRA for explaining fraud-classifier outputs. The model supports analyst review; it does not predict transactions independently, make customer decisions or prove fraud.

## Conda setup (CPU)

Use this optional CPU environment to read the code, run tests and chat without a GPU. For practical GPT-2 XL fine-tuning on the RTX 5060, follow [the device setup](DEVICE_GUIDE.md) and [CUDA installation guide](README_RTX5060.md) instead; do not replace its CUDA environment with this CPU setup.

```bash
conda env create -f environment.yml
conda activate fraudlm2
python -m pytest tests/test_guard.py tests/test_threshold_calculation.py -q
```

`environment.yml` installs Python 3.11 and a CPU build of PyTorch. For an NVIDIA GPU, use the `venv` steps below instead.

## Main commands

Run these from the repository root with `fraudlm2` active. **Run 1** is the original BAF adapter `xl_r16_v1`; **run 2** is `xl_r16_multidata_5060_v1`, continued on the outputs of eight classifiers.

| Step | Run 1 (BAF) | Run 2 (eight classifiers) |
|---|---|---|
| Build data | `python -m preprocess_data.fetch_public`, then `python -m preprocess_data.build_sft` (rewrites `dataset/`; see the script header) | `python preprocess_data/build_multisource_sft.py` (needs the private prediction files; see [RTX 5060 guide](README_RTX5060.md#prepare-refreshed-data)) |
| Train | `python training_model/train_sft.py --config <settings.json>`; the completed run used a Kaggle T4 ([runbook](kaggle/RUNBOOK_G4.md)) | RTX 5060 only; see [RTX 5060 guide](README_RTX5060.md#train-a-separate-refreshed-run) |
| Chat | `python chat.py --adapter checkpoint/xl_r16_v1/lora_adapter.pt` ([chat notes](README_chat_dev.md)) | `python chat_multisource.py --adapter checkpoint/xl_r16_multidata_5060_v1/lora_adapter.pt` ([chat guide](README_RTX5060.md#chat)) |
| Evaluate | `python evaluation/run_eval.py --size gpt2-xl --adapter checkpoint/xl_r16_v1/lora_adapter.pt --split test` ([notes](evaluation/EVAL_NOTES.md)) | `python evaluation/compare_original_validation.py --device cpu` ([grounding status](FINAL_GROUNDING.md)) |

For a small CPU training smoke check without an existing adapter (first use may download pretrained weights):

```bash
python training_model/train_sft.py --size gpt2-small --max_train_rows 32 --max_steps 10 --micro_batch 2 --grad_accum 1 --eval_every 5 --run_id smoke
```

**Run-1 demo chat.** The original adapter is not stored in Git (see [distribution status](DISTRIBUTION.md)). If you have an authorised private copy, place it at `checkpoint/xl_r16_v1/lora_adapter.pt`, next to its matching `run_config.json`; `chat.py` reads the model size from that configuration. Demo applications come from `preprocessed/ml_scores_months6_7_demo.csv`. GPT-2 XL downloads pretrained weights on first use. Its base weights alone take roughly 6 GB in fp32 or 3 GB in bf16; adapters, activations, caches and runtime add memory overhead. CPU responses can take seconds to minutes depending on hardware and token limits. Follow the device guide; CUDA is for the NVIDIA PC, not the Mac. Then type `list`, `explain <application id>` or `help`.

## Start here

- [This branch's device setup](DEVICE_GUIDE.md)
- [Development conventions](DEVELOPMENT.md)
- [RTX 5060 setup and commands](README_RTX5060.md)
- [Current grounding and evaluation status](FINAL_GROUNDING.md)
- [Data and model distribution status](DISTRIBUTION.md)
- [Original adapter evaluation](evaluation/EVAL_NOTES.md)

Open the repository folder in PyCharm. Install Python and a CUDA-compatible PyTorch build on the PC, then install `requirements-rtx5060.txt`. Run `python utils/check_cuda.py` before training.

## Project layout

| Path | Contents |
|---|---|
| `training_model/` | GPT-2 architecture, LoRA, supervised fine-tuning |
| `preprocess_data/` | Data preparation and multi-source SFT builder |
| `configs/` | Current prediction registry and training configurations |
| `utils/` | Prompt formatting, safety checks and private transfer packaging |
| `evaluation/` | Evaluation scripts, protocols and aggregate results |
| `chat_multisource.py` | Prediction lookup and guarded chat |
| `.run/` | Shared PyCharm run configurations |
| `checkpoint/` | Local weights and training state; not bundled in Git |
| `dataset_final_v2/` | Local refreshed training data and prediction store; Git-ignored |

## Current model

The completed RTX run is `xl_r16_multidata_5060_v1`: 6,016 steps and final validation loss 0.0589 on a mostly templated set. That loss must not be compared directly with the original adapter's loss on a different dataset. Small context checks do not establish improved generalisation or retention of all original skills.

The current registry uses one frozen classifier per dataset. Model facts are read at runtime, not recalled from adapter weights. The completed adapter still contains older training facts; it has not been retrained on the refreshed data. The separate `rtx5060_final_models.json` configuration is for a future run.

On the same original 500 validation rows, original/continued adapter losses are 0.327504/0.336809; on 422 overlap-screened rows they are 0.379331/0.390314. Lower is better, so this diagnostic does not show improvement. Full results and limitations are in [grounding status](FINAL_GROUNDING.md). Final code checks: 436 passed, one optional reference test skipped on the Mac.

## Sources and limitations

Architecture and training references include Sebastian Raschka's *Build a Large Language Model (From Scratch)* and the [companion repository](https://github.com/rasbt/LLMs-from-scratch). GPT-2 pretrained weights originate from OpenAI. Source, dataset and licence records remain in `dataset/SOURCES.md` and the existing project documentation.

The sources have different redistribution conditions. Private data transfer is not permission to redistribute. The continued model Release is temporarily unpublished pending review. Neither the adapter nor its rule-based guard is suitable for autonomous financial decisions.
