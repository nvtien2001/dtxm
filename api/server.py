# api/server.py
import os
import json
import tempfile
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse

from src.pipeline import TwoPassConfig, run_two_pass

app = FastAPI(title="DTXM API", version="1.0")


def _jsonable(obj):
    try:
        json.dumps(obj)
        return obj
    except Exception:
        if isinstance(obj, dict):
            return {str(k): _jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_jsonable(x) for x in obj]
        return str(obj)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/run")
async def run_pipeline(
    file: UploadFile = File(...),

    # --- Preview pass
    preview_max_edge: float = Form(1.0),
    preview_angle_deg: float = Form(25.0),
    preview_tolerance: float = Form(0.10),

    # --- Accurate pass
    accurate_max_edge: float = Form(0.5),
    accurate_angle_deg: float = Form(20.0),
    accurate_tolerance: float = Form(0.05),

    # --- Ollama
    use_ollama: bool = Form(True),
    ollama_model: str = Form(os.getenv("OLLAMA_MODEL", "llama3.2:3b")),
    ollama_url: str = Form(os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")),
    ollama_timeout_sec: int = Form(int(os.getenv("OLLAMA_TIMEOUT_SEC", "180"))),

    # --- Exposure
    exposure_rays: int = Form(int(os.getenv("EXPOSURE_RAYS", "256"))),
    exposure_samples_per_face: int = Form(int(os.getenv("EXPOSURE_SAMPLES_PER_FACE", "1"))),
    exposure_threshold_ratio: float = Form(float(os.getenv("EXPOSURE_THRESHOLD_RATIO", "0.5"))),
    exposure_soft_area: bool = Form(os.getenv("EXPOSURE_SOFT_AREA", "1") == "1"),
    exposure_max_points: int = Form(int(os.getenv("EXPOSURE_MAX_POINTS", "50000"))),
    exposure_seed: int = Form(int(os.getenv("EXPOSURE_SEED", "0"))),

    # --- Speed switches
    skip_preview_exposure: bool = Form(os.getenv("SKIP_PREVIEW_EXPOSURE", "1") == "1"),
    reuse_mesh_if_same_fingerprint: bool = Form(os.getenv("REUSE_MESH_IF_SAME", "1") == "1"),
    force_soft_area: bool = Form(os.getenv("FORCE_SOFT_AREA", "1") == "1"),

    # --- Debug
    debug_print_mesh_fingerprint: bool = Form(os.getenv("DEBUG_FINGERPRINT", "1") == "1"),
    debug_extreme_params_test: bool = Form(os.getenv("DEBUG_EXTREME", "0") == "1"),
):
    if not file.filename.lower().endswith(".3dm"):
        return JSONResponse({"ok": False, "error": "Only .3dm is supported"}, status_code=400)

    # save to temp
    suffix = ".3dm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
        tf.write(await file.read())
        tmp_path = tf.name

    cfg = TwoPassConfig(
        preview_max_edge=float(preview_max_edge),
        preview_angle_deg=float(preview_angle_deg),
        preview_tolerance=float(preview_tolerance),

        accurate_max_edge=float(accurate_max_edge),
        accurate_angle_deg=float(accurate_angle_deg),
        accurate_tolerance=float(accurate_tolerance),

        use_ollama=bool(use_ollama),
        ollama_model=str(ollama_model),
        ollama_url=str(ollama_url),
        ollama_timeout_sec=int(ollama_timeout_sec),

        exposure_rays=int(exposure_rays),
        exposure_samples_per_face=int(exposure_samples_per_face),
        exposure_threshold_ratio=float(exposure_threshold_ratio),
        exposure_soft_area=bool(exposure_soft_area),
        exposure_max_points=int(exposure_max_points),
        exposure_seed=int(exposure_seed),

        skip_preview_exposure=bool(skip_preview_exposure),
        reuse_mesh_if_same_fingerprint=bool(reuse_mesh_if_same_fingerprint),
        force_soft_area=bool(force_soft_area),

        debug_print_mesh_fingerprint=bool(debug_print_mesh_fingerprint),
        debug_extreme_params_test=bool(debug_extreme_params_test),
    )

    try:
        out = run_two_pass(tmp_path, cfg)
        out = _jsonable(out)
        return JSONResponse(out)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
