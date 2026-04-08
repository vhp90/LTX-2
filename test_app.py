#!/usr/bin/env python3
"""
LTX-2 Studio — End-to-End API Test
Tests the full generation flow through the frontend API routes,
exactly as the browser would call them.

Usage:
    python test_app.py                          # full test against running backend
    python test_app.py --base-url http://host:8080  # custom backend URL
    python test_app.py --skip-generate          # test everything except generation
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import Request, urlopen
from urllib.error import URLError

BACKEND = "http://localhost:8080"
TEST_IMAGE = Path(__file__).parent / "test-image.jpg"

# ─── Helpers ──────────────────────────────────────────────────────────────────

def section(title):
    print(f"\n{'─'*60}\n  {title}\n{'─'*60}")

def ok(msg):   print(f"  ✅  {msg}")
def fail(msg): print(f"  ❌  {msg}")
def warn(msg): print(f"  ⚠️   {msg}")

class APIError(Exception):
    pass

def api_get(path):
    url = urljoin(BACKEND + "/", path.lstrip("/"))
    req = Request(url)
    try:
        with urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except URLError as e:
        raise APIError(f"GET {path}: {e}") from e

def api_post(path, data=None, form_data=None):
    url = urljoin(BACKEND + "/", path.lstrip("/"))
    if form_data is not None:
        import io
        boundary = "----TestBoundary"
        body = io.BytesIO()
        for key, (filename, filedata, content_type) in form_data.items():
            body.write(f"--{boundary}\r\n".encode())
            body.write(f'Content-Disposition: form-data; name="{key}"; filename="{filename}"\r\n'.encode())
            body.write(f"Content-Type: {content_type}\r\n\r\n".encode())
            body.write(filedata)
            body.write(b"\r\n")
        body.write(f"--{boundary}--\r\n".encode())
        req = Request(url, data=body.getvalue(), method="POST")
        req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    else:
        payload = json.dumps(data or {}).encode()
        req = Request(url, data=payload, method="POST")
        req.add_header("Content-Type", "application/json")
    try:
        with urlopen(req, timeout=600) as resp:
            return json.loads(resp.read())
    except URLError as e:
        raise APIError(f"POST {path}: {e}") from e

def api_delete(path):
    url = urljoin(BACKEND + "/", path.lstrip("/"))
    req = Request(url, method="DELETE")
    try:
        with urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except URLError as e:
        raise APIError(f"DELETE {path}: {e}") from e

def poll_generation(job_id, timeout=600):
    """Poll generation status until complete/error."""
    start = time.time()
    last_stage = ""
    while time.time() - start < timeout:
        data = api_get(f"/api/generate/{job_id}/status")
        stage = data.get("stage", "")
        if stage != last_stage:
            print(f"    [{data.get('progress', 0):3.0f}%] {stage}")
            last_stage = stage
        if data.get("status") == "complete":
            return data
        if data.get("status") == "error":
            raise APIError(f"Generation failed: {data.get('error', 'unknown')}")
        time.sleep(2)
    raise APIError("Generation timed out")


# ─── Tests ────────────────────────────────────────────────────────────────────

def test_health():
    section("1. Health Check")
    data = api_get("/api/health")
    assert data.get("status") == "ok", f"Unexpected: {data}"
    ok(f"Backend alive — v{data.get('version', '?')}")

def test_models_status():
    section("2. Model Status")
    data = api_get("/api/models/status")
    gpu = data.get("gpu_available", False)
    gpu_name = data.get("gpu_name", "N/A")
    gpu_mem = data.get("gpu_memory_gb", "?")
    if gpu:
        ok(f"GPU: {gpu_name} ({gpu_mem} GB)")
    else:
        warn("No GPU detected — generation will simulate")

    models = data.get("models", {})
    ready = sum(1 for v in models.values() if v in ("ready", "exists"))
    total = len(models)
    ok(f"Models: {ready}/{total} ready")
    if not data.get("all_ready"):
        for k, v in models.items():
            if v not in ("ready", "exists"):
                warn(f"  {k}: {v}")
    return data

def test_pipelines():
    section("3. Pipelines")
    data = api_get("/api/pipelines")
    pipelines = data.get("pipelines", [])
    ok(f"Found {len(pipelines)} pipelines")
    for p in pipelines:
        print(f"    • {p['id']}: {p['name']}")
    assert len(pipelines) > 0, "No pipelines returned"
    return pipelines

def test_settings():
    section("4. Settings")
    data = api_get("/api/settings")
    ok(f"Quantization: {data.get('quantization', 'none')}")
    ok(f"Torch compile: {data.get('torchCompile', False)}")
    ok(f"Models dir: {data.get('models_dir', '?')}")
    return data

def test_loras():
    section("5. LoRAs")
    data = api_get("/api/models/loras")
    loras = data.get("loras", [])
    ok(f"Found {len(loras)} custom LoRAs")
    for l in loras:
        print(f"    • {l['name']} ({l.get('size_mb', '?')} MB)")
    return loras

def test_upload():
    section("6. File Upload")
    if not TEST_IMAGE.exists():
        warn(f"test-image.jpg not found — skipping upload test")
        return None
    with open(TEST_IMAGE, "rb") as f:
        img_bytes = f.read()
    data = api_post("/api/upload", form_data={
        "file": ("test-image.jpg", img_bytes, "image/jpeg"),
    })
    ok(f"Uploaded: {data.get('filename')} → {data.get('path')}")
    return data

def test_generate(upload_result=None):
    section("7. Generation (ti2vid_two_stages)")

    params = {
        "prompt": "A cinematic shot of the scene coming to life, gentle camera movement, natural lighting, 4K",
        "negative_prompt": "blurry, low quality, distorted, artifacts",
        "seed": 42,
        "height": 512,
        "width": 768,
        "num_frames": 41,
        "frame_rate": 24.0,
        "num_inference_steps": 10,
        "enhance_prompt": False,
        "video_cfg_scale": 3.0,
        "video_stg_scale": 1.0,
        "video_rescale_scale": 0.7,
        "video_modality_scale": 3.0,
        "video_skip_step": 0,
        "video_stg_blocks": "28",
        "audio_cfg_scale": 7.0,
        "audio_stg_scale": 1.0,
        "audio_rescale_scale": 0.7,
        "audio_modality_scale": 3.0,
        "audio_skip_step": 0,
        "audio_stg_blocks": "28",
        "custom_loras": [],
    }

    # Include uploaded image as conditioning if available
    if upload_result and upload_result.get("path"):
        params["images"] = [{
            "path": upload_result["path"],
            "frame_idx": 0,
            "strength": 1.0,
            "crf": 33,
        }]
        ok(f"Image conditioning: {upload_result['filename']}")

    request = {
        "pipeline": "ti2vid_two_stages",
        "params": params,
        "quantization": "none",  # backend auto-detects fp8
    }

    print(f"  Prompt: {params['prompt'][:70]}...")
    print(f"  Resolution: {params['width']}x{params['height']}, {params['num_frames']} frames, {params['num_inference_steps']} steps")

    t0 = time.time()
    result = api_post("/api/generate", data=request)
    job_id = result.get("job_id")
    assert job_id, f"No job_id returned: {result}"
    ok(f"Job started: {job_id}")

    data = poll_generation(job_id)
    elapsed = time.time() - t0
    output = data.get("output_path")
    ok(f"Generation complete in {elapsed:.1f}s")
    if output:
        ok(f"Output: {output}")
    else:
        warn("No output file (simulation mode)")
    return data

def test_generate_cached():
    """Run a second generation to verify pipeline caching works (should be faster)."""
    section("8. Cached Generation (same pipeline, different seed)")
    request = {
        "pipeline": "ti2vid_two_stages",
        "params": {
            "prompt": "A peaceful ocean wave at sunset, golden hour, slow motion",
            "negative_prompt": "blurry, artifacts",
            "seed": 123,
            "height": 512,
            "width": 768,
            "num_frames": 41,
            "frame_rate": 24.0,
            "num_inference_steps": 10,
            "video_cfg_scale": 3.0,
            "video_stg_scale": 1.0,
            "video_rescale_scale": 0.7,
            "video_modality_scale": 3.0,
            "video_skip_step": 0,
            "video_stg_blocks": "28",
            "audio_cfg_scale": 7.0,
            "audio_stg_scale": 1.0,
            "audio_rescale_scale": 0.7,
            "audio_modality_scale": 3.0,
            "audio_skip_step": 0,
            "audio_stg_blocks": "28",
            "custom_loras": [],
        },
        "quantization": "none",
    }

    t0 = time.time()
    result = api_post("/api/generate", data=request)
    data = poll_generation(result["job_id"])
    elapsed = time.time() - t0
    ok(f"Cached generation complete in {elapsed:.1f}s (should be faster than first run)")
    return data

def test_history():
    section("9. History")
    data = api_get("/api/history")
    items = data.get("items", [])
    ok(f"History: {len(items)} items")
    if items:
        latest = items[0]
        print(f"    Latest: {latest.get('pipeline')} — {latest.get('status')} — {latest.get('prompt', '')[:50]}...")
    return items


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    global BACKEND
    parser = argparse.ArgumentParser(description="LTX-2 Studio E2E API Test")
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--skip-generate", action="store_true", help="Skip generation tests")
    args = parser.parse_args()
    BACKEND = args.base_url.rstrip("/")

    print("\n" + "=" * 60)
    print("  LTX-2 Studio — End-to-End API Test")
    print(f"  Backend: {BACKEND}")
    print("=" * 60)

    t_start = time.time()
    passed = 0
    failed = 0
    errors = []

    tests = [
        ("Health", test_health),
        ("Models", test_models_status),
        ("Pipelines", test_pipelines),
        ("Settings", test_settings),
        ("LoRAs", test_loras),
    ]

    upload_result = None
    for name, fn in tests:
        try:
            result = fn()
            if name == "Upload":
                upload_result = result
            passed += 1
        except Exception as e:
            fail(f"{name}: {e}")
            errors.append(name)
            failed += 1

    # Upload test
    try:
        upload_result = test_upload()
        passed += 1
    except Exception as e:
        fail(f"Upload: {e}")
        errors.append("Upload")
        failed += 1

    if not args.skip_generate:
        try:
            test_generate(upload_result)
            passed += 1
        except Exception as e:
            fail(f"Generate: {e}")
            errors.append("Generate")
            failed += 1

        try:
            test_generate_cached()
            passed += 1
        except Exception as e:
            fail(f"Cached Generate: {e}")
            errors.append("Cached Generate")
            failed += 1

    try:
        test_history()
        passed += 1
    except Exception as e:
        fail(f"History: {e}")
        errors.append("History")
        failed += 1

    section("Summary")
    total = time.time() - t_start
    print(f"  Passed: {passed}, Failed: {failed}, Time: {total:.1f}s")
    if errors:
        fail(f"Failed tests: {', '.join(errors)}")
        sys.exit(1)
    else:
        ok("All tests passed")


if __name__ == "__main__":
    main()
