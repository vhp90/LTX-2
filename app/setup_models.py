#!/usr/bin/env python3
"""
LTX-2 Studio — Model Setup Script
Downloads all required models for the LTX-2 video generation pipeline.

Usage:
    python setup_models.py                          # Download all models
    python setup_models.py --dry-run                # Preview what would be downloaded
    python setup_models.py --models-dir ~/my-models # Custom download directory

Models are downloaded from HuggingFace using the `huggingface_hub` library.
Authentication is resolved in order:
    1. --hf-token CLI argument
    2. HF_TOKEN environment variable (Lightning.ai Secrets)
    3. Cached huggingface-cli login token
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("setup-models")

# ============================================================================
#  MODEL DEFINITIONS
# ============================================================================
# All models required for the LTX-2.3 pipeline
MODELS = {
    "checkpoint": {
        "name": "LTX-2.3 Checkpoint (22B)",
        "hf_repo": "Lightricks/LTX-2",
        "hf_filename": "ltx-2.3-22b-dev.safetensors",
        "local_subdir": "checkpoints",
        "size_approx": "~22 GB",
    },
    "distilled_lora": {
        "name": "Distilled LoRA (Stage 2)",
        "hf_repo": "Lightricks/LTX-2",
        "hf_filename": "ltx-2.3-22b-distilled-lora-384.safetensors",
        "local_subdir": "loras",
        "size_approx": "~400 MB",
    },
    "spatial_upsampler": {
        "name": "Spatial Upsampler (2×)",
        "hf_repo": "Lightricks/LTX-2",
        "hf_filename": "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        "local_subdir": "upsampler",
        "size_approx": "~100 MB",
    },
    "gemma_encoder": {
        "name": "Gemma 3 Text Encoder (4B-IT)",
        "hf_repo": "google/gemma-3-4b-it",
        "is_full_repo": True,
        "local_subdir": "gemma",
        "size_approx": "~8 GB",
    },
}


# ============================================================================
#  TOKEN RESOLUTION
# ============================================================================
def resolve_hf_token(cli_token: str | None = None) -> str | None:
    """Resolve HuggingFace token from CLI > env > cached login."""
    # 1. CLI argument
    if cli_token:
        logger.info("Using HF token from CLI argument")
        return cli_token

    # 2. Environment variable
    env_token = os.environ.get("HF_TOKEN")
    if env_token:
        logger.info("Using HF token from HF_TOKEN environment variable")
        return env_token

    # 3. Cached huggingface-cli login
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        if api.token:
            logger.info("Using cached HF token from huggingface-cli login")
            return api.token
    except Exception:
        pass

    return None


# ============================================================================
#  DOWNLOAD FUNCTIONS
# ============================================================================
def download_single_file(
    repo_id: str,
    filename: str,
    local_dir: Path,
    token: str | None,
    dry_run: bool = False,
) -> bool:
    """Download a single file from a HuggingFace repo."""
    local_path = local_dir / filename

    if local_path.exists() and local_path.stat().st_size > 1000:
        logger.info(f"  ✅ Already exists: {local_path}")
        return True

    if dry_run:
        logger.info(f"  📋 Would download: {repo_id}/{filename} → {local_path}")
        return True

    try:
        from huggingface_hub import hf_hub_download

        logger.info(f"  ⬇️  Downloading: {repo_id}/{filename}")
        local_dir.mkdir(parents=True, exist_ok=True)

        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(local_dir),
            token=token,
        )
        logger.info(f"  ✅ Downloaded: {local_path}")
        return True

    except Exception as e:
        logger.error(f"  ❌ Failed: {e}")
        return False


def download_full_repo(
    repo_id: str,
    local_dir: Path,
    token: str | None,
    dry_run: bool = False,
) -> bool:
    """Download a full HuggingFace repo (e.g., Gemma text encoder)."""
    repo_name = repo_id.split("/")[-1]
    local_path = local_dir / repo_name

    if local_path.exists() and any(local_path.iterdir()):
        logger.info(f"  ✅ Already exists: {local_path}")
        return True

    if dry_run:
        logger.info(f"  📋 Would download: {repo_id} → {local_path}")
        return True

    try:
        from huggingface_hub import snapshot_download

        logger.info(f"  ⬇️  Downloading repo: {repo_id}")
        local_path.parent.mkdir(parents=True, exist_ok=True)

        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_path),
            token=token,
        )
        logger.info(f"  ✅ Downloaded: {local_path}")
        return True

    except Exception as e:
        logger.error(f"  ❌ Failed: {e}")
        return False


# ============================================================================
#  MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Download required models for LTX-2 Studio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=str(Path.home() / "ltx-models"),
        help="Base directory for model storage (default: ~/ltx-models)",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace API token (can also use HF_TOKEN env var)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be downloaded without downloading",
    )
    parser.add_argument(
        "--only",
        type=str,
        nargs="+",
        choices=list(MODELS.keys()),
        default=None,
        help="Only download specific models",
    )
    args = parser.parse_args()

    base_dir = Path(args.models_dir).expanduser()

    print()
    print("=" * 60)
    print("  LTX-2 Studio — Model Setup")
    print("=" * 60)
    print(f"  Models directory: {base_dir}")
    print(f"  Dry run: {args.dry_run}")
    print()

    # Resolve token
    token = resolve_hf_token(args.hf_token)
    if not token and not args.dry_run:
        logger.warning(
            "No HuggingFace token found. Some models may require authentication.\n"
            "Set HF_TOKEN in environment or run: huggingface-cli login"
        )

    # Download models
    models_to_download = args.only or list(MODELS.keys())
    results = {}
    total = len(models_to_download)

    for idx, key in enumerate(models_to_download, 1):
        info = MODELS[key]
        print(f"[{idx}/{total}] {info['name']} ({info['size_approx']})")

        local_dir = base_dir / info["local_subdir"]

        if info.get("is_full_repo"):
            success = download_full_repo(
                repo_id=info["hf_repo"],
                local_dir=local_dir,
                token=token,
                dry_run=args.dry_run,
            )
        else:
            success = download_single_file(
                repo_id=info["hf_repo"],
                filename=info["hf_filename"],
                local_dir=local_dir,
                token=token,
                dry_run=args.dry_run,
            )

        results[key] = "ok" if success else "failed"
        print()

    # Summary
    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    for key, status in results.items():
        icon = "✅" if status == "ok" else "❌"
        print(f"  {icon} {MODELS[key]['name']}: {status}")

    failed = [k for k, v in results.items() if v != "ok"]
    if failed:
        print(f"\n  ⚠️  {len(failed)} model(s) failed to download.")
        sys.exit(1)
    else:
        print(f"\n  🎉 All {total} models ready!")
        print(f"  📁 Location: {base_dir}")
    print()


if __name__ == "__main__":
    main()
