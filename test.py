# test.py
import os
import json
import time

def load_env_if_any():
    # optional .env support
    try:
        from dotenv import load_dotenv
        load_dotenv()
        return True
    except Exception:
        return False

def main():
    load_env_if_any()

    # --- IMPORTANT: set BREPMESH_BINDIR for pipeline import ---
    # If you already set it in .env, no need to set here.
    # os.environ.setdefault("BREPMESH_BINDIR", r"D:\Workspace\DTXM\brepmesh\build\out\Release")

    from src.pipeline import TwoPassConfig, run_two_pass  # import after env

    path = os.getenv("DTXM_INPUT_3DM", r"D:\Workspace\DTXM\models\GVPFDDW000001.500.3dm")

    cfg = TwoPassConfig(
        # Coarse first to avoid huge meshes for runtime mesher Phase 1
        preview_max_edge=float(os.getenv("PREVIEW_MAX_EDGE", "5.0")),
        preview_angle_deg=float(os.getenv("PREVIEW_ANGLE_DEG", "25")),
        preview_tolerance=float(os.getenv("PREVIEW_TOL", "0.10")),

        accurate_max_edge=float(os.getenv("ACCURATE_MAX_EDGE", "2.0")),
        accurate_angle_deg=float(os.getenv("ACCURATE_ANGLE_DEG", "20")),
        accurate_tolerance=float(os.getenv("ACCURATE_TOL", "0.05")),

        # Ollama
        use_ollama=True,
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat"),
        ollama_timeout_sec=int(os.getenv("OLLAMA_TIMEOUT_SEC", "180")),

        # Exposure (stub for now)
        exposure_rays=int(os.getenv("EXPOSURE_RAYS", "2000")),
        exposure_samples_per_face=int(os.getenv("EXPOSURE_SAMPLES_PER_FACE", "1")),
        exposure_threshold_ratio=float(os.getenv("EXPOSURE_THRESHOLD_RATIO", "0.5")),
    )

    print("=== DTXM TEST (two-pass + ollama) ===", flush=True)
    print("Input 3DM:", path, flush=True)
    print("BREPMESH_BINDIR:", os.getenv("BREPMESH_BINDIR"), flush=True)
    print("OLLAMA_URL:", cfg.ollama_url, flush=True)
    print("OLLAMA_MODEL:", cfg.ollama_model, flush=True)
    print("Preview params:", cfg.preview_max_edge, cfg.preview_angle_deg, cfg.preview_tolerance, flush=True)
    print("Accurate params:", cfg.accurate_max_edge, cfg.accurate_angle_deg, cfg.accurate_tolerance, flush=True)
    print("------------------------------------", flush=True)

    t0 = time.time()
    try:
        out = run_two_pass(path, cfg)
    except Exception as e:
        print("RUN FAILED:", type(e).__name__, str(e), flush=True)
        raise

    dt = time.time() - t0
    print("DONE in %.3fs" % dt, flush=True)

    # Summary
    print("\n=== PREVIEW ===", flush=True)
    print("nv/nt:", out["preview"]["nv"], out["preview"]["nt"], flush=True)
    print("bbox_diag:", out["preview"]["metrics"]["bbox_diag"], flush=True)

    print("\n=== OLLAMA ===", flush=True)
    print("ollama_error:", out["ollama_error"], flush=True)
    plan = out.get("plan")
    if plan is None:
        print("plan: None", flush=True)
    else:
        print("plan:", json.dumps(plan, ensure_ascii=False, indent=2), flush=True)

    print("\n=== ACCURATE ===", flush=True)
    print("meshing:", out["accurate"]["meshing"], flush=True)
    print("nv/nt:", out["accurate"]["nv"], out["accurate"]["nt"], flush=True)

    # Save artifact
    out_path = "out_ollama_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\nSaved:", out_path, flush=True)

if __name__ == "__main__":
    main()
