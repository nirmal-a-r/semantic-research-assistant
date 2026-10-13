"""Download every model used by the retrieval experiments into the local cache.

Run once while the network is available; afterwards the experiment scripts can run
offline (HF_HUB_OFFLINE=1).

Usage:
    python scripts/prefetch_models.py
"""

import os
import time

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import snapshot_download

WEIGHTS = ["*.json", "*.txt", "*.safetensors", "*.model", "spm.model", "spiece*",
           "merges*", "vocab*", "1_Pooling/*"]

MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "cross-encoder/ms-marco-MiniLM-L6-v2",
    "cross-encoder/ms-marco-MiniLM-L12-v2",
    "google/flan-t5-small",
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-small-en-v1.5",
    "intfloat/e5-base-v2",
    "FacebookAI/roberta-base",
]


def fetch(repo, attempts=5):
    for attempt in range(1, attempts + 1):
        t0 = time.time()
        try:
            path = snapshot_download(repo_id=repo, allow_patterns=WEIGHTS, max_workers=2)
            size = sum(f.stat().st_size for f in __import__("pathlib").Path(path).rglob("*") if f.is_file())
            print(f"OK   {repo:45s} {size/1e6:7.0f} MB in {time.time()-t0:5.0f}s", flush=True)
            return True
        except Exception as exc:
            print(f"RETRY {attempt}/{attempts} {repo}: {type(exc).__name__}", flush=True)
            time.sleep(5 * attempt)
    print(f"FAIL {repo}", flush=True)
    return False


def main():
    ok = [fetch(m) for m in MODELS]
    print(f"PREFETCH_DONE {sum(ok)}/{len(MODELS)} models cached", flush=True)


if __name__ == "__main__":
    main()
