"""Fetch the DFlash2 drafter for single-user mode (SPEC=dflash2).

The default is the pinned W4A16-GPTQ release artifact (about 1.2 GB instead of
the 3.85 GB BF16 source checkpoint).

  venv/bin/python prepare/fetch_dflash2.py [dst_dir]         # default models/Qwen3.8-27B-DFlash2-W4A16
  venv/bin/python prepare/fetch_dflash2.py --bf16 [dst_dir]  # the original bf16 drafter instead (incoai)
"""
import os, sys
from huggingface_hub import snapshot_download

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
BF16 = "--bf16" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
REPO = "incoai/Qwen3.8-27B-DFlash2" if BF16 else "syvai/Qwen3.8-27B-DFlash2-W4A16"
REVISION = None if BF16 else "4d30ec736ffc6b8688dc2ae2b502d9b48bdec279"
D = args[0] if args else os.path.join(ROOT, "models", "Qwen3.8-27B-DFlash2" + ("" if BF16 else "-W4A16"))
os.makedirs(D, exist_ok=True)
snapshot_download(REPO, revision=REVISION, local_dir=D,
                  allow_patterns=["*.json", "*.safetensors", "README.md"])
print("drafter ready:", D)
print("the single-user service uses this drafter automatically")
