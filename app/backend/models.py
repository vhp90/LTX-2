"""
LTX-2 Studio — Model Manager
Reads app/config.yaml and downloads models directly from pasted URLs.
Handles huggingface_hub and pyyaml installation dynamically.
Uses Lightning.ai secrets (HF_TOKEN, CIVITAI_TOKEN) from env.
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urlparse, unquote
from typing import Any

logger = logging.getLogger(__name__)

def ensure_dependencies():
    missing = []
    try:
        import huggingface_hub
    except ImportError:
        missing.append("huggingface_hub")
    
    try:
        import yaml
    except ImportError:
        missing.append("pyyaml")

    if missing:
        logger.info(f"Installing missing dependencies: {', '.join(missing)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
        import site
        from importlib import reload
        reload(site)

ensure_dependencies()

import yaml
from huggingface_hub import hf_hub_download, snapshot_download

from app.backend.config import AppSettings, APP_DIR


def get_filename_from_url_and_headers(url: str, headers: dict | None = None) -> str:
    """Extract filename from Content-Disposition header, falling back to URL path."""
    if headers and (cd := headers.get("Content-Disposition") or headers.get("content-disposition")):
        m = re.search(r"filename\*=(?:UTF-8''|utf-8'')(.+?)(?:;|$)", cd, re.I)
        if m:
            return unquote(m.group(1).strip().strip('"'))
        m = re.search(r'filename="?([^";]+)', cd, re.I)
        if m:
            return m.group(1).strip().strip('"')

    # Fallback to URL path
    path = urlparse(url).path
    name = Path(unquote(path)).name
    if not name or name.isdigit():  # Like CivitAI's /models/12345
        return f"model_{hash(url)}.safetensors"
    return name


def fetch_civitai_filename(url: str) -> str:
    """Do a HEAD request to civitai to get the actual filename."""
    try:
        req = urllib.request.Request(url, method="HEAD")
        token = os.environ.get("CIVITAI_TOKEN")
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(req) as response:
            headers = dict(response.getheaders())
            return get_filename_from_url_and_headers(url, headers)
    except Exception:
        return get_filename_from_url_and_headers(url)


class ModelManager:
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.config_path = APP_DIR / "config.yaml"

    def get_yaml_config(self) -> dict:
        if not self.config_path.exists():
            return {"models": {}}
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f) or {}

    def fetch_models_list(self) -> list[dict]:
        config_data = self.get_yaml_config()
        base_dir = Path(config_data.get("settings", {}).get("models_dir", self.settings.models_dir)).expanduser()
        
        models_list = []
        for category, urls in config_data.get("models", {}).items():
            if not urls:
                continue
            for url in urls:
                if not isinstance(url, str) or not url.strip():
                    continue
                url = url.strip()
                path_parts = [p for p in urlparse(url).path.split("/") if p]
                
                info = {
                    "category": category,
                    "url": url,
                    "local_dir": base_dir / category,
                    "is_directory": False,
                }
                
                if "huggingface.co" in url:
                    if "resolve" in path_parts or "blob" in path_parts:
                        idx = path_parts.index("resolve") if "resolve" in path_parts else path_parts.index("blob")
                        info["hf_repo"] = "/".join(path_parts[:idx])
                        info["hf_file"] = "/".join(path_parts[idx+2:])
                        info["filename"] = path_parts[-1]
                    else:
                        info["hf_repo"] = "/".join(path_parts[-2:])
                        info["is_directory"] = True
                        info["filename"] = path_parts[-1]
                else:
                    # Non-HF (like CivitAI) - determine name dynamically during download,
                    # but for now give it a placeholder or fetched name if not cached
                    # Actually, if it already exists in the dir matching the expected CivExt, we find it later.
                    info["filename"] = get_filename_from_url_and_headers(url)
                    
                models_list.append(info)
                
        return models_list

    def check_status(self) -> dict[str, Any]:
        """Check status of models. For generic URLs, counts items in category dir."""
        result = {"models": {}, "all_ready": True}
        models_list = self.fetch_models_list()
        
        for info in models_list:
            if "huggingface.co" in info["url"]:
                key = info["filename"]
                local_path = info["local_dir"] / info["filename"]
                is_ready = (local_path.exists() and any(local_path.iterdir())) if info["is_directory"] else (local_path.exists() and local_path.stat().st_size > 1000)
                result["models"][key] = "ready" if is_ready else "missing"
                if not is_ready:
                    result["all_ready"] = False
            else:
                # Generic URL - just check if the directory has generic safetensors, 
                # but to be strict, we'd need the real filename.
                # If we haven't downloaded it, we'll mark as missing.
                # It's better to resolve filename via HEAD once and hold it, 
                # but since check_status is lightweight, let's keep it simple:
                key = f"{urlparse(info['url']).hostname}_{hash(info['url']) % 10000}"
                result["models"][key] = "ready" if info["local_dir"].exists() and len(list(info["local_dir"].glob("*.safetensors"))) > 0 else "missing"

        try:
            import torch
            result["gpu_available"] = torch.cuda.is_available()
            if result["gpu_available"]:
                result["gpu_name"] = torch.cuda.get_device_name(0)
                result["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        except ImportError:
            # torch not installed yet — fall back to nvidia-smi for GPU detection
            result["gpu_available"] = False
            try:
                import subprocess as _sp
                out = _sp.check_output(["nvidia-smi", "--query-gpu=name,memory.total",
                                        "--format=csv,noheader,nounits"], text=True).strip()
                if out:
                    parts = out.split(",")
                    result["gpu_available"] = True
                    result["gpu_name"] = parts[0].strip()
                    result["gpu_memory_gb"] = round(float(parts[1].strip()) / 1024, 1)
            except Exception:
                pass

        return result

    def download_all(self, hf_token: str | None = None, on_progress: callable | None = None) -> dict[str, str]:
        token = hf_token or os.environ.get("HF_TOKEN")
        civitai_token = os.environ.get("CIVITAI_TOKEN")
        models_list = self.fetch_models_list()
        results = {}
        total = len(models_list)

        for idx, info in enumerate(models_list):
            category = info["category"]
            local_dir = info["local_dir"]
            progress_base = (idx / total) * 100

            # For non-HF URLs (e.g. CivitAI): check if ANY safetensors already
            # exists in the category dir BEFORE doing a slow network HEAD request.
            if "huggingface.co" not in info["url"]:
                existing = list(local_dir.glob("*.safetensors")) if local_dir.exists() else []
                # Match by hash embedded in the placeholder filename we generated earlier
                placeholder = info["filename"]  # e.g. model_<hash>.safetensors
                matched = next((f for f in existing if f.name == placeholder), None)
                if matched is None and existing:
                    # Filename not yet resolved — check if dir has enough files
                    # (one per URL in this category). Count URLs vs existing files.
                    urls_in_cat = sum(
                        1 for m in models_list
                        if m["category"] == category and "huggingface.co" not in m["url"]
                    )
                    if len(existing) >= urls_in_cat:
                        # All files for this category already present, skip HEAD
                        key = placeholder
                        results[key] = "exists"
                        if on_progress: on_progress(key, progress_base + (100 / total), f"{key}: Already exists")
                        continue
                elif matched:
                    results[placeholder] = "exists"
                    if on_progress: on_progress(placeholder, progress_base + (100 / total), f"{placeholder}: Already exists")
                    continue
                # Need to download — resolve real filename via HEAD now
                filename = fetch_civitai_filename(info["url"])
                info["filename"] = filename
            else:
                filename = info["filename"]

            local_path = local_dir / filename

            if info["is_directory"]:
                if local_path.exists() and any(local_path.iterdir()):
                    results[filename] = "exists"
                    if on_progress: on_progress(filename, progress_base + (100 / total), f"{filename}: Already exists")
                    continue
            else:
                if local_path.exists() and local_path.stat().st_size > 1000:
                    results[filename] = "exists"
                    if on_progress: on_progress(filename, progress_base + (100 / total), f"{filename}: Already exists")
                    continue

            if on_progress: on_progress(filename, progress_base, f"Downloading {filename}...")

            try:
                local_dir.mkdir(parents=True, exist_ok=True)
                
                if "huggingface.co" in info["url"]:
                    if info["is_directory"]:
                        snapshot_download(repo_id=info["hf_repo"], local_dir=str(local_path), token=token)
                    else:
                        hf_hub_download(repo_id=info["hf_repo"], filename=info["hf_file"], local_dir=str(local_dir), token=token)
                else:
                    req = urllib.request.Request(info["url"])
                    req.add_header("User-Agent", "Mozilla/5.0")
                    if civitai_token and "civitai.com" in info["url"]:
                        req.add_header("Authorization", f"Bearer {civitai_token}")
                        
                    with urllib.request.urlopen(req) as response:
                        # Get real filename from response headers (Content-Disposition)
                        resp_headers = dict(response.getheaders()) if hasattr(response, 'getheaders') else {}
                        real_name = get_filename_from_url_and_headers(info["url"], resp_headers)
                        if real_name and real_name != filename:
                            filename = real_name
                            info["filename"] = real_name
                            local_path = local_dir / real_name

                        with open(local_path, 'wb') as out_file:
                            while True:
                                chunk = response.read(65536)
                                if not chunk:
                                    break
                                out_file.write(chunk)

                results[filename] = "downloaded"
            except Exception as e:
                logger.error(f"Download failed for {info['url']}: {e}")
                results[filename] = f"error: {str(e)}"
                
            if on_progress:
                status = "Done" if results.get(filename) in ("downloaded", "exists") else "Failed"
                on_progress(filename, progress_base + (100 / total), f"{filename}: {status}")

        return results

    def get_model_paths(self) -> dict[str, str]:
        paths = {}
        config_data = self.get_yaml_config()
        base_dir = Path(config_data.get("settings", {}).get("models_dir", self.settings.models_dir)).expanduser()
        
        for category, urls in config_data.get("models", {}).items():
            if not urls: continue
            cat_dir = base_dir / category
            
            if category == "checkpoints":
                for file in cat_dir.glob("*.safetensors"):
                    paths["checkpoint"] = str(file)
                    break
            elif category == "upsampler":
                for file in cat_dir.glob("*.safetensors"):
                    paths["spatial_upsampler"] = str(file)
                    break
            elif category == "loras":
                paths["loras_dir"] = str(cat_dir)
                for file in cat_dir.glob("*distilled-lora*.safetensors"):
                    paths["distilled_lora"] = str(file)
                    break
            elif category == "gemma":
                dirs = [d for d in cat_dir.iterdir() if d.is_dir()]
                if dirs: paths["gemma"] = str(dirs[0])
                    
        return paths

    def get_available_loras(self) -> list[dict]:
        """Return a list of available LoRAs."""
        paths = self.get_model_paths()
        loras_dir = paths.get("loras_dir")
        if not loras_dir:
            return []
            
        loras = []
        dir_path = Path(loras_dir)
        if dir_path.exists():
            for file in dir_path.glob("*.safetensors"):
                # We skip the distilled lora as it's automatically handled by the pipeline
                if "distilled" in file.name:
                    continue
                # Try to read trigger word from safetensors metadata
                trigger_word = ""
                try:
                    from safetensors import safe_open
                    with safe_open(str(file), framework="pt") as f:
                        meta = f.metadata() or {}
                        # Common metadata keys for trigger words
                        trigger_word = (
                            meta.get("ss_trigger_words", "") or
                            meta.get("trigger_word", "") or
                            meta.get("activation_text", "") or
                            meta.get("modelspec.trigger_phrase", "") or
                            ""
                        )
                except Exception:
                    pass
                loras.append({
                    "name": file.stem,  # Use stem (no extension) for cleaner display
                    "path": str(file),
                    "size_mb": round(file.stat().st_size / 1024 / 1024, 1),
                    "trigger_word": trigger_word,
                })
        return loras
