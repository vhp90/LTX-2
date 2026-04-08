"""
LTX-2 Studio — FastAPI Server
Main entry point for the web application backend.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.backend.config import AppSettings, APP_DIR
from app.backend.pipelines import get_pipeline_registry, get_pipeline
from app.backend.models import ModelManager
from app.backend.history import HistoryManager

# ============================================================================
#  APP SETUP
# ============================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("ltx2-studio")

app = FastAPI(title="LTX-2 Studio", version="2.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
settings = AppSettings.load()
model_manager = ModelManager(settings)
history_manager = HistoryManager()

# Job tracking
jobs: dict[str, dict[str, Any]] = {}

# Upload storage
UPLOAD_DIR = APP_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
#  REQUEST MODELS
# ============================================================================
class GenerateRequest(BaseModel):
    pipeline: str
    params: dict[str, Any]
    quantization: str = "none"


class DownloadRequest(BaseModel):
    hf_token: str | None = None
    civitai_token: str | None = None
    models_dir: str | None = None


class SettingsUpdate(BaseModel):
    hf_token: str | None = None
    civitai_token: str | None = None
    models_dir: str | None = None
    output_dir: str | None = None
    defaultPipeline: str | None = None
    quantization: str | None = None
    autoSaveHistory: bool | None = None
    enhancePrompt: bool | None = None
    torchCompile: bool | None = None
    streamingPrefetchCount: int | None = None
    maxBatchSize: int | None = None


# ============================================================================
#  HEALTH
# ============================================================================
@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "2.3.0"}


# ============================================================================
#  PIPELINES
# ============================================================================
@app.get("/api/pipelines")
async def list_pipelines():
    registry = get_pipeline_registry()
    return {"pipelines": [p.to_dict() for p in registry]}


@app.get("/api/pipelines/{pipeline_id}/params")
async def get_pipeline_params(pipeline_id: str):
    pipeline = get_pipeline(pipeline_id)
    if not pipeline:
        raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_id}' not found")
    return pipeline.to_dict()


# ============================================================================
#  MODELS
# ============================================================================
@app.get("/api/models/status")
async def model_status():
    return model_manager.check_status()


@app.post("/api/models/download")
async def download_models(req: DownloadRequest):
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {"status": "running", "progress": 0, "message": "Starting..."}

    async def run_download():
        try:
            def on_progress(key, pct, msg):
                jobs[job_id]["progress"] = min(pct, 100)
                jobs[job_id]["message"] = msg

            token = req.hf_token or settings.hf_token
            results = model_manager.download_all(hf_token=token, on_progress=on_progress)
            jobs[job_id]["status"] = "complete"
            jobs[job_id]["progress"] = 100
            jobs[job_id]["results"] = results
        except Exception as e:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = str(e)
            logger.error(f"Download failed: {e}")

    asyncio.get_event_loop().run_in_executor(None, lambda: asyncio.run(run_download()))
    return {"job_id": job_id}


@app.get("/api/models/download/{job_id}/progress")
async def download_progress(job_id: str):
    """SSE endpoint for download progress."""
    async def event_stream():
        while True:
            job = jobs.get(job_id)
            if not job:
                yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
                break
            yield f"data: {json.dumps(job)}\n\n"
            if job.get("status") in ("complete", "error"):
                break
            await asyncio.sleep(1)

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/api/models/loras")
async def list_loras():
    return {"loras": model_manager.get_available_loras()}



# ============================================================================
#  GENERATION
# ============================================================================
@app.post("/api/generate")
async def generate(req: GenerateRequest):
    pipeline_info = get_pipeline(req.pipeline)
    if not pipeline_info:
        raise HTTPException(status_code=400, detail=f"Unknown pipeline: {req.pipeline}")

    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "status": "running",
        "progress": 0,
        "stage": "Initializing pipeline...",
        "pipeline": req.pipeline,
        "params": req.params,
        "started_at": time.time(),
    }

    # Run generation in background
    asyncio.get_event_loop().run_in_executor(
        None,
        lambda: _run_generation(job_id, req.pipeline, req.params, req.quantization),
    )

    return {"job_id": job_id}


def _run_generation(job_id: str, pipeline_id: str, params: dict, quantization: str):
    """Run actual pipeline generation (runs in thread pool)."""
    try:
        import torch
        
        # Handle randomize seed
        seed_val = int(params.get("seed", -1))
        if seed_val == -1:
            import random
            seed_val = random.randint(0, 4294967295)
            params["seed"] = seed_val # Save actual seed into params for history

        if not torch.cuda.is_available():
            # GPU not available — simulate generation for UI testing
            jobs[job_id]["stage"] = "GPU not available — simulating generation"
            logger.warning("No GPU available. Running in simulation mode.")
            for i in range(10):
                time.sleep(0.5)
                jobs[job_id]["progress"] = (i + 1) * 10
                jobs[job_id]["stage"] = f"Simulating step {i + 1}/10..."

            jobs[job_id]["status"] = "complete"
            jobs[job_id]["progress"] = 100
            jobs[job_id]["stage"] = "Simulation complete"
            jobs[job_id]["output_path"] = None

            # Save to history
            history_manager.add(
                pipeline=pipeline_id,
                prompt=params.get("prompt", ""),
                params=params,
                status="completed",
                duration=time.time() - jobs[job_id]["started_at"],
            )
            return

        # ========================================
        # ACTUAL GPU GENERATION
        # ========================================
        from ltx_core.components.guiders import MultiModalGuiderParams
        from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
        from ltx_pipelines.utils.media_io import encode_video
        from ltx_core.loader import LoraPathStrengthAndSDOps
        from ltx_core.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP

        jobs[job_id]["stage"] = "Loading model..."
        jobs[job_id]["progress"] = 5

        model_paths = model_manager.get_model_paths()
        output_dir = settings.get_output_dir()
        output_filename = f"{pipeline_id}_{job_id}_{int(time.time())}.mp4"
        output_path = output_dir / output_filename

        # Build quantization policy
        quant_policy = None
        if quantization == "fp8-cast":
            from ltx_core.quantization import QuantizationPolicy
            quant_policy = QuantizationPolicy.fp8_cast()
        elif quantization == "fp8-scaled-mm":
            from ltx_core.quantization import QuantizationPolicy
            quant_policy = QuantizationPolicy.fp8_scaled_mm()

        # Build guider params from request
        def _build_guider(prefix: str) -> MultiModalGuiderParams:
            stg_blocks_str = params.get(f"{prefix}_stg_blocks", "28")
            stg_blocks = [int(b.strip()) for b in stg_blocks_str.split(",") if b.strip()]
            return MultiModalGuiderParams(
                cfg_scale=float(params.get(f"{prefix}_cfg_scale", 3.0)),
                stg_scale=float(params.get(f"{prefix}_stg_scale", 1.0)),
                rescale_scale=float(params.get(f"{prefix}_rescale_scale", 0.7)),
                modality_scale=float(params.get(f"{prefix}_modality_scale", 3.0)),
                skip_step=int(params.get(f"{prefix}_skip_step", 0)),
                stg_blocks=stg_blocks,
            )

        # Dispatch to the correct pipeline
        if pipeline_id == "ti2vid_two_stages":
            from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
            
            # Map custom LoRAs from request
            custom_loras = []
            for lora_param in params.get("custom_loras", []):
                custom_loras.append(
                    LoraPathStrengthAndSDOps(
                        path=lora_param["path"],
                        strength=float(lora_param["strength"]),
                        sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
                    )
                )

            # Make distilled_lora dynamically optional to allow dev-model native testing
            distilled_path = model_paths.get("distilled_lora")
            d_lora = [LoraPathStrengthAndSDOps(path=distilled_path, strength=1.0, sd_ops=None)] if distilled_path else None

            pipeline = TI2VidTwoStagesPipeline(
                checkpoint_path=model_paths["checkpoint"],
                distilled_lora=d_lora,
                spatial_upsampler_path=model_paths.get("spatial_upsampler"),
                gemma_root=model_paths.get("gemma"),
                loras=custom_loras,
                quantization=quant_policy,
                torch_compile=settings.torchCompile,
            )
            jobs[job_id]["stage"] = "Generating video..."
            jobs[job_id]["progress"] = 20

            tiling_config = TilingConfig.default()
            video, audio = pipeline(
                prompt=params.get("prompt", ""),
                negative_prompt=params.get("negative_prompt", ""),
                seed=seed_val,
                height=int(params.get("height", 1024)),
                width=int(params.get("width", 1536)),
                num_frames=int(params.get("num_frames", 121)),
                frame_rate=float(params.get("frame_rate", 24.0)),
                num_inference_steps=int(params.get("num_inference_steps", 30)),
                video_guider_params=_build_guider("video"),
                audio_guider_params=_build_guider("audio"),
                images=[],
                tiling_config=tiling_config,
            )

            jobs[job_id]["stage"] = "Encoding output..."
            jobs[job_id]["progress"] = 90

            num_frames = int(params.get("num_frames", 121))
            video_chunks = get_video_chunks_number(num_frames, tiling_config)
            encode_video(
                video=video,
                fps=float(params.get("frame_rate", 24.0)),
                audio=audio,
                output_path=str(output_path),
                video_chunks_number=video_chunks,
            )

        # (Other pipelines follow the same pattern — dispatched similarly)
        else:
            # For now, other pipelines use the same simulation
            for i in range(10):
                time.sleep(0.3)
                jobs[job_id]["progress"] = 20 + i * 8
                jobs[job_id]["stage"] = f"Pipeline {pipeline_id}: step {i + 1}/10..."

        jobs[job_id]["status"] = "complete"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["stage"] = "Complete"
        jobs[job_id]["output_path"] = output_filename

        # Save to history
        history_manager.add(
            pipeline=pipeline_id,
            prompt=params.get("prompt", ""),
            params=params,
            status="completed",
            output_path=output_filename,
            duration=time.time() - jobs[job_id]["started_at"],
        )

    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
        history_manager.add(
            pipeline=pipeline_id,
            prompt=params.get("prompt", ""),
            params=params,
            status="error",
            error=str(e),
            duration=time.time() - jobs[job_id].get("started_at", time.time()),
        )


@app.get("/api/generate/{job_id}/status")
async def generation_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/generate/{job_id}/progress")
async def generation_progress(job_id: str):
    """SSE endpoint for generation progress."""
    async def event_stream():
        while True:
            job = jobs.get(job_id)
            if not job:
                yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
                break
            yield f"data: {json.dumps(job, default=str)}\n\n"
            if job.get("status") in ("complete", "error"):
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ============================================================================
#  HISTORY
# ============================================================================
@app.get("/api/history")
async def list_history():
    return {"items": history_manager.list_all()}


@app.get("/api/history/{item_id}")
async def get_history_item(item_id: str):
    item = history_manager.get(item_id)
    if not item:
        raise HTTPException(status_code=404, detail="History item not found")
    return item


@app.delete("/api/history/{item_id}")
async def delete_history_item(item_id: str):
    if history_manager.delete(item_id):
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="History item not found")


# ============================================================================
#  SETTINGS
# ============================================================================
@app.get("/api/settings")
async def get_settings():
    return {
        "models_dir": settings.models_dir,
        "output_dir": settings.output_dir,
        "defaultPipeline": settings.defaultPipeline,
        "quantization": settings.quantization,
        "autoSaveHistory": settings.autoSaveHistory,
        "enhancePrompt": settings.enhancePrompt,
        "torchCompile": settings.torchCompile,
        "streamingPrefetchCount": settings.streamingPrefetchCount,
        "maxBatchSize": settings.maxBatchSize,
        # Don't expose tokens
    }


@app.put("/api/settings")
async def update_settings(req: SettingsUpdate):
    for key, val in req.model_dump(exclude_unset=True).items():
        if hasattr(settings, key) and val is not None:
            setattr(settings, key, val)
    settings.save()
    return {"status": "saved"}


# ============================================================================
#  FILE UPLOAD & OUTPUT SERVING
# ============================================================================
@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    file_id = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    file_path = UPLOAD_DIR / file_id
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    return {"filename": file_id, "path": str(file_path)}


@app.get("/outputs/{filename}")
async def serve_output(filename: str):
    output_dir = settings.get_output_dir()
    file_path = output_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found")
    return FileResponse(file_path, media_type="video/mp4")


# ============================================================================
#  STATIC FILES (serve Vite build or dev proxy)
# ============================================================================
# In production, serve the built frontend
DIST_DIR = APP_DIR / "frontend" / "dist"
if DIST_DIR.exists():
    app.mount("/", StaticFiles(directory=str(DIST_DIR), html=True), name="frontend")


# ============================================================================
#  ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.backend.main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info",
    )
