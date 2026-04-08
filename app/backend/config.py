"""
LTX-2 Studio — Application Configuration
Central configuration with model paths, defaults, and settings persistence.
"""
import json
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict

# ============================================================================
#  PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # LTX-2/
APP_DIR = PROJECT_ROOT / "app"
BACKEND_DIR = APP_DIR / "backend"

DEFAULT_MODELS_DIR = Path.home() / "ltx-models"
DEFAULT_OUTPUT_DIR = Path.home() / "ltx-outputs"
SETTINGS_FILE = APP_DIR / "settings.json"
HISTORY_FILE = APP_DIR / "history.json"

# ============================================================================
#  MODELS
# ============================================================================
# All required models for the LTX-2.3 pipeline
REQUIRED_MODELS = {
    "checkpoint": {
        "name": "LTX-2.3 Checkpoint",
        "filename": "ltx-2.3-22b-dev.safetensors",
        "hf_repo": "Lightricks/LTX-2",
        "hf_path": "ltx-2.3-22b-dev.safetensors",
        "subfolder": "checkpoints",
        "size_gb": 22,
    },
    "distilled_lora": {
        "name": "Distilled LoRA",
        "filename": "ltx-2.3-22b-distilled-lora-384.safetensors",
        "hf_repo": "Lightricks/LTX-2",
        "hf_path": "ltx-2.3-22b-distilled-lora-384.safetensors",
        "subfolder": "loras",
        "size_gb": 0.4,
    },
    "spatial_upsampler": {
        "name": "Spatial Upsampler",
        "filename": "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        "hf_repo": "Lightricks/LTX-2",
        "hf_path": "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        "subfolder": "upsampler",
        "size_gb": 0.1,
    },
    "gemma": {
        "name": "Gemma 3 Text Encoder",
        "hf_repo": "google/gemma-3-4b-it",
        "subfolder": "gemma",
        "is_directory": True,
        "size_gb": 8,
    },
}


# ============================================================================
#  SETTINGS
# ============================================================================
@dataclass
class AppSettings:
    """Application settings with defaults from the LTX-2 pipeline constants."""
    # Tokens
    hf_token: str = ""
    civitai_token: str = ""

    # Paths
    models_dir: str = str(DEFAULT_MODELS_DIR)
    output_dir: str = str(DEFAULT_OUTPUT_DIR)

    # Defaults
    defaultPipeline: str = "ti2vid_two_stages"
    quantization: str = "none"
    autoSaveHistory: bool = True
    enhancePrompt: bool = False
    torchCompile: bool = False
    streamingPrefetchCount: int | None = None
    maxBatchSize: int = 1

    def save(self) -> None:
        """Persist settings to disk."""
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        # Don't persist tokens to disk for security
        data.pop("hf_token", None)
        data.pop("civitai_token", None)
        with open(SETTINGS_FILE, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls) -> "AppSettings":
        """Load settings from disk + env vars."""
        settings = cls()

        # Load from file
        if SETTINGS_FILE.exists():
            with open(SETTINGS_FILE) as f:
                data = json.load(f)
            for key, val in data.items():
                if hasattr(settings, key):
                    setattr(settings, key, val)

        # Tokens from environment (Lightning.ai Secrets)
        settings.hf_token = os.environ.get("HF_TOKEN", settings.hf_token)
        settings.civitai_token = os.environ.get("CIVITAI_TOKEN", settings.civitai_token)

        return settings

    def get_model_path(self, model_key: str) -> Path:
        """Get the full path for a required model."""
        model_info = REQUIRED_MODELS.get(model_key)
        if not model_info:
            raise ValueError(f"Unknown model key: {model_key}")

        base = Path(self.models_dir).expanduser()
        subfolder = model_info.get("subfolder", "")

        if model_info.get("is_directory"):
            return base / subfolder / model_info["hf_repo"].split("/")[-1]
        return base / subfolder / model_info["filename"]

    def get_output_dir(self) -> Path:
        """Get the output directory, creating it if needed."""
        path = Path(self.output_dir).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path
