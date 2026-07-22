# Sulphur-2 IC-LoRA Training Bootstrap & Recovery Guide

This guide details how to launch, monitor, and seamlessly recover the Sulphur-2 In-Context LoRA (IC-LoRA) training run in a new or restarted Lightning.ai Studio environment.

---

## Single-Command Bootstrap & Resume

To launch or resume training in any environment (including a fresh Studio restart):

```bash
uv run python scripts/supervisor.py --config projects/v2v-ic-lora/config.yaml
```

### What the Supervisor Does Automatically

1. **Discovers State**: Checks local directory (`projects/v2v-ic-lora/outputs/checkpoints`) and the remote Hugging Face repository (`ahp93/lorarun`) for the latest complete checkpoint.
2. **Auto-Restores**: Downloads the newest verified checkpoint and training state from Hugging Face if local state is absent.
3. **Supervises Trainer**: Spawns `packages/ltx-trainer/scripts/train.py` with `config.yaml`, streaming stdout/stderr to both the console and timestamped log files under `projects/v2v-ic-lora/logs/`.
4. **Monitors Health**: Collects GPU VRAM, temperature, GPU utilization, RAM, and step loss telemetry.
5. **Atomic Remote Sync**: After every completed checkpoint (every 250 steps), uploads an atomic commit to `ahp93/lorarun` containing:
   - Checkpoint weights (`lora_weights_step_XXXXX.safetensors`)
   - Optimizer & scheduler state (`training_state_step_XXXXX.pt`)
   - Exact training configuration (`config.yaml`)
   - Run state manifest (`run_manifest.json`)
   - Base model manifest (`models/model_manifest.json`)
   - Log snapshot

---

## Base Model Lock & Verification

- **Base Checkpoint**: `SulphurAI/Sulphur-2-base` (`models/sulphur_dev_bf16.safetensors`)
- **SHA-256 Hash**: `330d6c138eebeeaf6c420ae50d346dec8ceb0f873ef1760aa5a26fe06b0e3918`
- **Text Encoder**: Gemma-3 12B (`models/gemma-3-12b-it`)

---

## Dataset Layout & Preprocessing

The full 72-pair dataset is stored in `data/dataset.json` (66 portrait pairs 544×960, 6 landscape pairs 960×544).

Preprocess the full dataset before full training:

```bash
uv run python packages/ltx-trainer/scripts/process_dataset.py data/dataset.json \
  --resolution-buckets "544x960x81;960x544x81" \
  --model-path models/sulphur_dev_bf16.safetensors \
  --text-encoder-path models/gemma-3-12b-it \
  --lora-trigger "Clothesoff" \
  --skip-audio \
  --output-dir projects/v2v-ic-lora/.precomputed
```
