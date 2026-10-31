#!/usr/bin/env python3
"""Offline dependency, model shard, and actual CUDA kernel preflight."""
import argparse
import importlib.metadata
import json
import struct
from pathlib import Path


def check(model_root, vlm_name="Qwen3-VL-8B-Instruct"):
    import torch
    # Package metadata can survive a deleted editable-install source directory.
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    report = {"packages": {}, "models": {}, "cuda": {}}
    report["sam3_importable"] = True
    for name in ("torch", "torchvision", "transformers", "accelerate", "sam3", "numpy", "scipy", "plyfile"):
        report["packages"][name] = importlib.metadata.version(name)
    for name in (vlm_name, "sam3"):
        directory = model_root / name
        files = sorted(directory.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"No model weights: {directory}")
        for path in files:
            with path.open("rb") as stream:
                length = struct.unpack("<Q", stream.read(8))[0]
                header = json.loads(stream.read(length))
            end = max(v["data_offsets"][1] for k, v in header.items() if k != "__metadata__")
            if 8 + length + end != path.stat().st_size:
                raise ValueError(f"Truncated or inconsistent weights: {path}")
        index = directory / "model.safetensors.index.json"
        if index.exists():
            expected = set(json.loads(index.read_text())["weight_map"].values())
            if not expected.issubset({p.name for p in files}):
                raise FileNotFoundError(f"Missing checkpoint shards: {expected - {p.name for p in files}}")
        report["models"][name] = {"safetensors_files": len(files), "structurally_complete": True}
    if not (model_root / "sam3/sam3.pt").is_file():
        raise FileNotFoundError("Official SAM3 point-prompt backend requires models/sam3/sam3.pt")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    # nvidia-smi alone does not establish that PyTorch supports the GPU.
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        a = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        assert torch.isfinite(a @ a).all()
        conv = torch.nn.Conv2d(3, 8, 3).cuda()
        assert torch.isfinite(conv(torch.randn(1, 3, 32, 32, device="cuda"))).all()
    torch.cuda.synchronize()
    report["cuda"] = {"device": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
                      "runtime": torch.version.cuda, "bf16_matmul": True, "convolution": True}
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=Path("models"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--vlm-name", default="Qwen3-VL-8B-Instruct")
    args = parser.parse_args()
    text = json.dumps(check(args.model_root, args.vlm_name), indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
