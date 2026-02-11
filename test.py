# test.py
import os
import json
import time


def load_env_if_any():
    try:
        from dotenv import load_dotenv
        load_dotenv()
        return True
    except Exception:
        return False


def _print_exposure_block(exposure_obj: dict, label: str):
    """
    In exposure an toàn, kể cả khi bị skip.
    """
    if not isinstance(exposure_obj, dict):
        print(f"{label} exposure: <invalid>", flush=True)
        return

    if exposure_obj.get("skipped", False):
        print(f"{label} exposure: SKIPPED ({exposure_obj.get('reason', '')})", flush=True)
        return

    # bình thường
    exposed_area = exposure_obj.get("exposed_area")
    exposed_ratio = exposure_obj.get("exposed_ratio")
    points = exposure_obj.get("points")
    rays = exposure_obj.get("rays")
    accel = exposure_obj.get("accel")

    print(f"{label} exposure: area={exposed_area} ratio={exposed_ratio} points={points} rays={rays} accel={accel}", flush=True)

    # nếu có stats mới
    if "avg_rays_used" in exposure_obj:
        print(
            f"{label} exposure stats: avg_rays_used={exposure_obj.get('avg_rays_used')}, "
            f"early_stop_ratio={exposure_obj.get('early_stop_ratio')}",
            flush=True
        )


def main():
    load_env_if_any()

    # Nếu bạn dùng .env thì set trong đó:
    # BREPMESH_BINDIR=D:\Workspace\DTXM\brepmesh\build\out\Release
    # DTXM_INPUT_3DM=D:\Workspace\DTXM\models\GVPFDDW000001.500.3dm

    from src.pipeline import TwoPassConfig, run_two_pass

    path = os.getenv("DTXM_INPUT_3DM", r"D:\Workspace\DTXM\models\GVPFDDW000001.500.3dm")

    cfg = TwoPassConfig(
        # Pass 1
        preview_max_edge=float(os.getenv("PREVIEW_MAX_EDGE", "1.0")),
        preview_angle_deg=float(os.getenv("PREVIEW_ANGLE_DEG", "25")),
        preview_tolerance=float(os.getenv("PREVIEW_TOL", "0.10")),

        # Pass 2 defaults (Ollama có thể override)
        accurate_max_edge=float(os.getenv("ACCURATE_MAX_EDGE", "0.5")),
        accurate_angle_deg=float(os.getenv("ACCURATE_ANGLE_DEG", "20")),
        accurate_tolerance=float(os.getenv("ACCURATE_TOL", "0.05")),

        # Ollama
        use_ollama=(os.getenv("USE_OLLAMA", "1") == "1"),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
        ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat"),
        ollama_timeout_sec=int(os.getenv("OLLAMA_TIMEOUT_SEC", "180")),

        # Exposure defaults
        exposure_rays=int(os.getenv("EXPOSURE_RAYS", "256")),
        exposure_samples_per_face=int(os.getenv("EXPOSURE_SAMPLES_PER_FACE", "1")),
        exposure_threshold_ratio=float(os.getenv("EXPOSURE_THRESHOLD_RATIO", "0.5")),
        exposure_soft_area=(os.getenv("EXPOSURE_SOFT_AREA", "1") == "1"),
        exposure_max_points=int(os.getenv("EXPOSURE_MAX_POINTS", "50000")),
        exposure_seed=int(os.getenv("EXPOSURE_SEED", "0")),

        # speed switches (đang dùng trong pipeline mới)
        skip_preview_exposure=(os.getenv("SKIP_PREVIEW_EXPOSURE", "1") == "1"),
        reuse_mesh_if_same_fingerprint=(os.getenv("REUSE_MESH_IF_SAME", "1") == "1"),
        force_soft_area=(os.getenv("FORCE_SOFT_AREA", "1") == "1"),

        # debug
        debug_print_mesh_fingerprint=(os.getenv("DEBUG_FINGERPRINT", "1") == "1"),
        debug_extreme_params_test=(os.getenv("DEBUG_EXTREME", "1") == "1"),
    )

    print("=== DTXM TEST (two-pass + ollama + exposed area) ===", flush=True)
    print("Input 3DM:", path, flush=True)
    print("BREPMESH_BINDIR:", os.getenv("BREPMESH_BINDIR"), flush=True)
    print("OLLAMA_URL:", cfg.ollama_url, flush=True)
    print("OLLAMA_MODEL:", cfg.ollama_model, flush=True)
    print("Preview params:", cfg.preview_max_edge, cfg.preview_angle_deg, cfg.preview_tolerance, flush=True)
    print("Accurate params:", cfg.accurate_max_edge, cfg.accurate_angle_deg, cfg.accurate_tolerance, flush=True)
    print("Exposure:", cfg.exposure_rays, cfg.exposure_samples_per_face, cfg.exposure_threshold_ratio, cfg.exposure_soft_area, flush=True)
    print("------------------------------------", flush=True)

    t0 = time.time()
    out = run_two_pass(path, cfg)
    dt = time.time() - t0
    print("DONE in %.3fs" % dt, flush=True)

    # --- PREVIEW
    print("\n=== PREVIEW ===", flush=True)
    print("nv/nt:", out["preview"]["nv"], out["preview"]["nt"], flush=True)
    print("bbox_diag:", out["preview"]["metrics"]["bbox_diag"], flush=True)
    print("total_area:", out["preview"]["surface"]["total_area"], flush=True)
    _print_exposure_block(out["preview"]["exposure"], "preview")

    # --- OLLAMA
    print("\n=== OLLAMA ===", flush=True)
    print("ollama_error:", out["ollama_error"], flush=True)
    plan = out.get("plan")
    if plan is None:
        print("plan: None", flush=True)
    else:
        print("plan:", json.dumps(plan, ensure_ascii=False, indent=2), flush=True)

    # --- ACCURATE
    print("\n=== ACCURATE ===", flush=True)
    print("meshing:", out["accurate"]["meshing"], flush=True)
    print("nv/nt:", out["accurate"]["nv"], out["accurate"]["nt"], flush=True)
    print("total_area:", out["accurate"]["surface"]["total_area"], flush=True)
    _print_exposure_block(out["accurate"]["exposure"], "accurate")

    out_path = "out_ollama_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\nSaved:", out_path, flush=True)


if __name__ == "__main__":
    main()
