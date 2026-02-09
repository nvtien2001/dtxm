from src.pipeline import TwoPassConfig, run_two_pass

path = r"D:\Workspace\DTXM\models\GVPFDDW000001.500.3dm"
cfg = TwoPassConfig(use_ollama=False)   # <- quan trọng
out = run_two_pass(path, cfg)
print("DONE", out["preview"]["nv"], out["preview"]["nt"])
