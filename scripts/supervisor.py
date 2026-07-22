#!/usr/bin/env python3

"""
Spot-Safe Supervisor for LTX-2 IC-LoRA Training.
Supervises training process, records telemetry, and backs up completed checkpoints
to Hugging Face Hub (ahp93/lorarun) atomically for seamless resume across Studio restarts.
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from huggingface_hub import HfApi, upload_folder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Supervisor")

HF_REPO_ID = "ahp93/lorarun"

def get_git_commit_sha():
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"

def compute_sha256(filepath):
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()

def get_gpu_telemetry():
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        parts = [p.strip() for p in res.stdout.strip().split(",")]
        return {
            "gpu_util_pct": float(parts[0]),
            "vram_used_mb": float(parts[1]),
            "vram_total_mb": float(parts[2]),
            "temp_c": float(parts[3])
        }
    except Exception as e:
        return {"error": str(e)}

class TrainingSupervisor:
    def __init__(self, config_path: str, poll_interval: int = 30):
        self.config_path = Path(config_path).resolve()
        self.run_dir = self.config_path.parent
        self.outputs_dir = self.run_dir / "outputs"
        self.checkpoints_dir = self.outputs_dir / "checkpoints"
        self.logs_dir = self.run_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        self.manifest_path = self.run_dir / "run_manifest.json"
        self.poll_interval = poll_interval
        
        self.token = os.environ.get("HF_TOKEN")
        if not self.token:
            raise ValueError("HF_TOKEN environment variable is required. Please run: export HF_TOKEN='your_token'")
        self.api = HfApi(token=self.token)

    def load_manifest(self):
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                return json.load(f)
        return {
            "run_id": self.run_dir.name,
            "git_commit": get_git_commit_sha(),
            "latest_local_checkpoint": None,
            "latest_remote_checkpoint": None,
            "uploaded_checkpoints": []
        }

    def save_manifest(self, manifest):
        with open(self.manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    def discover_and_restore_checkpoint(self, manifest):
        """Check local and remote for newest completed checkpoint to configure auto-resume."""
        local_ckpts = []
        if self.checkpoints_dir.exists():
            for p in self.checkpoints_dir.glob("lora_weights_step_*.safetensors"):
                if p.stat().st_size > 0:
                    try:
                        step = int(p.stem.split("step_")[1])
                        local_ckpts.append((step, p))
                    except ValueError:
                        pass
        
        latest_step = -1
        latest_ckpt_path = None

        if local_ckpts:
            latest_step, latest_ckpt_path = max(local_ckpts, key=lambda x: x[0])
            logger.info(f"Found local complete checkpoint at step {latest_step}: {latest_ckpt_path}")

        # Check remote repo if local is missing or behind
        try:
            remote_files = self.api.list_repo_files(repo_id=HF_REPO_ID)
            remote_ckpts = [f for f in remote_files if "lora_weights_step_" in f and f.endswith(".safetensors")]
            if remote_ckpts:
                remote_steps = []
                for f in remote_ckpts:
                    try:
                        s = int(Path(f).stem.split("step_")[1])
                        remote_steps.append((s, f))
                    except ValueError:
                        pass
                if remote_steps:
                    max_remote_step, max_remote_file = max(remote_steps, key=lambda x: x[0])
                    logger.info(f"Found remote checkpoint at step {max_remote_step} on {HF_REPO_ID}")
                    if max_remote_step > latest_step:
                        logger.info(f"Downloading newer remote checkpoint (step {max_remote_step})...")
                        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
                        local_target = self.checkpoints_dir / Path(max_remote_file).name
                        dl_ckpt = self.api.hf_hub_download(repo_id=HF_REPO_ID, filename=max_remote_file, local_dir=str(self.checkpoints_dir))
                        if Path(dl_ckpt) != local_target and Path(dl_ckpt).exists():
                            shutil.move(str(dl_ckpt), str(local_target))
                        
                        state_file = max_remote_file.replace("lora_weights_step_", "training_state_step_").replace(".safetensors", ".pt")
                        if state_file in remote_files:
                            dl_state = self.api.hf_hub_download(repo_id=HF_REPO_ID, filename=state_file, local_dir=str(self.checkpoints_dir))
                            target_state = self.checkpoints_dir / Path(state_file).name
                            if Path(dl_state) != target_state and Path(dl_state).exists():
                                shutil.move(str(dl_state), str(target_state))
                        
                        latest_step = max_remote_step
                        latest_ckpt_path = local_target
        except Exception as e:
            logger.info(f"No existing remote checkpoints found on {HF_REPO_ID} ({e})")

        if latest_ckpt_path:
            logger.info(f"Configuring config.yaml to resume from step {latest_step}: {latest_ckpt_path}")
            self.update_config_resume(latest_ckpt_path)
            manifest["latest_local_checkpoint"] = str(latest_ckpt_path)
            manifest["latest_step"] = latest_step
            self.save_manifest(manifest)
            return True
        return False

    def update_config_resume(self, checkpoint_path):
        import yaml
        with open(self.config_path) as f:
            cfg = yaml.safe_load(f)
        cfg["model"]["load_checkpoint"] = str(checkpoint_path)
        with open(self.config_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)

    def sync_checkpoint_to_hf(self, ckpt_path: Path, manifest: dict):
        step_str = ckpt_path.stem.split("step_")[1]
        step = int(step_str)
        
        if step in manifest.get("uploaded_checkpoints", []):
            return

        state_path = ckpt_path.parent / f"training_state_step_{step_str}.pt"
        if not ckpt_path.exists() or ckpt_path.stat().st_size == 0:
            logger.warning(f"Checkpoint file {ckpt_path} is incomplete. Postponing upload.")
            return

        logger.info(f"Uploading step {step} checkpoint to Hugging Face repository {HF_REPO_ID}...")

        # Create staging upload directory for atomic commit
        stage_dir = self.run_dir / "hf_staging"
        if stage_dir.exists():
            shutil.rmtree(stage_dir)
        stage_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(ckpt_path, stage_dir / ckpt_path.name)
        if state_path.exists():
            shutil.copy2(state_path, stage_dir / state_path.name)
        shutil.copy2(self.config_path, stage_dir / "config.yaml")
        if not self.manifest_path.exists():
            self.save_manifest(manifest)
        shutil.copy2(self.manifest_path, stage_dir / "run_manifest.json")

        model_manifest = Path("models/model_manifest.json")
        if model_manifest.exists():
            shutil.copy2(model_manifest, stage_dir / "model_manifest.json")

        for retries in range(3):
            try:
                self.api.upload_folder(
                    folder_path=str(stage_dir),
                    repo_id=HF_REPO_ID,
                    commit_message=f"Upload IC-LoRA checkpoint step {step} [Git: {manifest['git_commit'][:7]}]",
                    token=self.token
                )
                logger.info(f"✅ Successfully uploaded step {step} checkpoint to {HF_REPO_ID}!")
                manifest.setdefault("uploaded_checkpoints", []).append(step)
                manifest["latest_remote_checkpoint"] = f"lora_weights_step_{step_str}.safetensors"
                self.save_manifest(manifest)
                break
            except Exception as e:
                logger.error(f"HF upload attempt {retries+1} failed: {e}")
                time.sleep(5 * (retries + 1))
        
        if stage_dir.exists():
            shutil.rmtree(stage_dir)

    def run(self):
        logger.info("Initializing Spot-Safe Supervisor...")
        manifest = self.load_manifest()
        self.discover_and_restore_checkpoint(manifest)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = self.logs_dir / f"training_{timestamp}.log"
        logger.info(f"Logging training run output to {log_file_path}")

        cmd = ["uv", "run", "python", "packages/ltx-trainer/scripts/train.py", str(self.config_path)]
        logger.info(f"Launching training command: {' '.join(cmd)}")

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        with open(log_file_path, "a") as log_f:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(Path.cwd()),
                env=env
            )

            try:
                last_sync_time = time.time()
                while True:
                    line = process.stdout.readline()
                    if line:
                        sys.stdout.write(line)
                        sys.stdout.flush()
                        log_f.write(line)
                        log_f.flush()
                    elif process.poll() is not None:
                        break

                    # Periodically check for new checkpoints to upload (every 10s)
                    now = time.time()
                    if now - last_sync_time > 10:
                        last_sync_time = now
                        if self.checkpoints_dir.exists():
                            for ckpt in self.checkpoints_dir.glob("lora_weights_step_*.safetensors"):
                                self.sync_checkpoint_to_hf(ckpt, manifest)

                rc = process.returncode
                logger.info(f"Trainer process exited with code {rc}")
                
                # Final sync of checkpoints
                if self.checkpoints_dir.exists():
                    for ckpt in self.checkpoints_dir.glob("lora_weights_step_*.safetensors"):
                        self.sync_checkpoint_to_hf(ckpt, manifest)

                return rc

            except KeyboardInterrupt:
                logger.warning("Received termination signal. Stopping trainer gracefully...")
                process.terminate()
                process.wait(timeout=10)
                return 130

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spot-Safe Supervisor for LTX-2 IC-LoRA Training")
    parser.add_argument("--config", required=True, help="Path to training config.yaml")
    args = parser.parse_args()

    supervisor = TrainingSupervisor(config_path=args.config)
    sys.exit(supervisor.run())
