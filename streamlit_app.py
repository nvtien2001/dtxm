# ui/streamlit_app.py
import os
import json
import tempfile
import streamlit as st

from src.pipeline import TwoPassConfig, run_two_pass


def _jsonable(obj):
    # output của pipeline chủ yếu là dict/list/float/int.
    try:
        json.dumps(obj)
        return obj
    except Exception:
        if isinstance(obj, dict):
            return {str(k): _jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_jsonable(x) for x in obj]
        return str(obj)


def main():
    st.set_page_config(page_title="DTXM - 3DM Pipeline", layout="wide")
    st.title("Import .3dm → Ollama → JSON")

    with st.sidebar:
        st.header("Cấu hình")

        # --- Preview pass
        # st.subheader("Pass 1 (preview)")
        # preview_max_edge = st.number_input("preview_max_edge", value=1.0, min_value=0.001, step=0.1, format="%.6f")
        # preview_angle_deg = st.number_input("preview_angle_deg", value=25.0, min_value=0.0, step=1.0, format="%.3f")
        # preview_tolerance = st.number_input("preview_tolerance", value=0.10, min_value=0.0, step=0.01, format="%.6f")

        # # --- Accurate pass
        # st.subheader("Pass 2 (accurate defaults)")
        # accurate_max_edge = st.number_input("accurate_max_edge", value=0.5, min_value=0.001, step=0.1, format="%.6f")
        # accurate_angle_deg = st.number_input("accurate_angle_deg", value=20.0, min_value=0.0, step=1.0, format="%.3f")
        # accurate_tolerance = st.number_input("accurate_tolerance", value=0.05, min_value=0.0, step=0.01, format="%.6f")

        # --- Ollama
        st.subheader("Ollama")
        use_ollama = st.checkbox("use_ollama", value=True)
        ollama_model = st.text_input("ollama_model", value=os.getenv("OLLAMA_MODEL", "llama3.2:3b"))
        ollama_url = st.text_input("ollama_url", value=os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat"))
        ollama_timeout_sec = st.number_input("ollama_timeout_sec", value=int(os.getenv("OLLAMA_TIMEOUT_SEC", "180")), min_value=1, step=10)

        # --- Exposure
        st.subheader("Exposure defaults")
        exposure_rays = st.number_input("exposure_rays", value=int(os.getenv("EXPOSURE_RAYS", "256")), min_value=1, step=32)
        exposure_samples_per_face = st.number_input("exposure_samples_per_face", value=int(os.getenv("EXPOSURE_SAMPLES_PER_FACE", "1")), min_value=1, step=1)
        exposure_threshold_ratio = st.number_input("exposure_threshold_ratio", value=float(os.getenv("EXPOSURE_THRESHOLD_RATIO", "0.5")), min_value=0.0, max_value=1.0, step=0.05, format="%.3f")
        exposure_soft_area = st.checkbox("exposure_soft_area", value=(os.getenv("EXPOSURE_SOFT_AREA", "1") == "1"))
        exposure_max_points = st.number_input("exposure_max_points", value=int(os.getenv("EXPOSURE_MAX_POINTS", "50000")), min_value=1000, step=5000)
        exposure_seed = st.number_input("exposure_seed", value=int(os.getenv("EXPOSURE_SEED", "0")), min_value=0, step=1)

        # --- Speed switches
        st.subheader("Speed switches")
        skip_preview_exposure = st.checkbox("skip_preview_exposure", value=(os.getenv("SKIP_PREVIEW_EXPOSURE", "1") == "1"))
        reuse_mesh_if_same_fingerprint = st.checkbox("reuse_mesh_if_same_fingerprint", value=(os.getenv("REUSE_MESH_IF_SAME", "1") == "1"))
        force_soft_area = st.checkbox("force_soft_area", value=(os.getenv("FORCE_SOFT_AREA", "1") == "1"))

        # --- Debug
        st.subheader("Debug")
        debug_print_mesh_fingerprint = st.checkbox("debug_print_mesh_fingerprint", value=(os.getenv("DEBUG_FINGERPRINT", "1") == "1"))
        debug_extreme_params_test = st.checkbox("debug_extreme_params_test", value=(os.getenv("DEBUG_EXTREME", "0") == "1"))

    # st.info("Lưu ý: nếu không import được brepmesh (.pyd), hãy set env `BREPMESH_BINDIR=.../brepmesh/build/out/Release` trước khi chạy UI.")

    up = st.file_uploader("Upload file .3dm", type=["3dm"])
    if not up:
        st.stop()

    col1, col2 = st.columns([1, 1])
    with col1:
        run_btn = st.button("Chạy pipeline", type="primary")
    with col2:
        st.write("")

    if not run_btn:
        st.stop()

    # save uploaded file to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=".3dm") as tf:
        tf.write(up.getbuffer())
        tmp_path = tf.name

    cfg = TwoPassConfig(
        # preview_max_edge=float(preview_max_edge),
        # preview_angle_deg=float(preview_angle_deg),
        # preview_tolerance=float(preview_tolerance),

        # accurate_max_edge=float(accurate_max_edge),
        # accurate_angle_deg=float(accurate_angle_deg),
        # accurate_tolerance=float(accurate_tolerance),

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
        with st.spinner("Đang chạy pipeline..."):
            out = run_two_pass(tmp_path, cfg)
        out = _jsonable(out)

        st.success("DONE")
        st.subheader("Kết quả JSON")
        st.json(out, expanded=True)

        json_bytes = json.dumps(out, ensure_ascii=False, indent=2).encode("utf-8")
        st.download_button(
            "Tải JSON",
            data=json_bytes,
            file_name="dtxm_result.json",
            mime="application/json",
        )
    except Exception as e:
        st.error(f"Lỗi: {e}")
        st.code(str(e), language="text")
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


if __name__ == "__main__":
    main()
