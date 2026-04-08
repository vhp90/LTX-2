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

# Add project root to path (one directory up from app/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("setup-models")

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
        description="Download required models for LTX-2 Web Studio natively from config.yaml",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models-dir", type=str, default=None,
        help="Override base directory for model storage",
    )
    parser.add_argument(
        "--hf-token", type=str, default=None,
        help="HuggingFace API token (can also use HF_TOKEN env var)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview what would be downloaded without downloading",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  LTX-2 Studio — Model Setup (Directly targeting config.yaml)")
    print("=" * 60)

    # Resolve token securely
    token = resolve_hf_token(args.hf_token)
    if not token and not args.dry_run:
        logger.warning(
            "No HuggingFace token found. Some models may require authentication.\n"
            "Set HF_TOKEN in environment or run: huggingface-cli login"
        )

    try:
        from app.backend.config import AppSettings
        from app.backend.models import ModelManager
    except ImportError as e:
        logger.error(f"Failed to import app modules: {e}")
        sys.exit(1)

    settings = AppSettings()
    # Optionally override models_dir
    if args.models_dir:
        settings.models_dir = args.models_dir

    manager = ModelManager(settings)
    yaml_config = manager.get_yaml_config()
    
    print(f"  Configuration Loaded: {manager.config_path}")
    print(f"  Base Directory: {Path(yaml_config.get('settings', {}).get('models_dir', settings.models_dir)).expanduser()}")
    print(f"  Dry run active: {args.dry_run}\n")

    if args.dry_run:
        models = manager.fetch_models_list()
        for idx, m in enumerate(models, 1):
            print(f"[{idx}] Would Download: {m['url']} -> {m['local_dir'].name}")
        print("\n  📋 Dry-run complete.")
        return

    # Trigger actual native pipeline sweep
    def _print_progress(filename, pct, status_msg):
        print(f"\r  [{pct:3.0f}%] {status_msg}", end="", flush=True)

    print("Initiating direct config.yaml synchronization...")
    results = manager.download_all(hf_token=token, on_progress=_print_progress)
    print("\n\n" + "=" * 60)
    print("  Synchronization Summary")
    print("=" * 60)
    
    failed = 0
    for name, result in results.items():
        if result in ("downloaded", "exists"):
            print(f"  ✅ {name}: {result}")
        else:
            print(f"  ❌ {name}: {result}")
            failed += 1

    if failed:
        sys.exit(1)

if __name__ == "__main__":
    main()
