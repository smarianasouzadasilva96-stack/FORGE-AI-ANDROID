"""Export own trained PyTorch Transformer weights to the native Android runtime."""
import argparse
import struct
from pathlib import Path
import torch


def export(checkpoint, output):
    data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config, state = data["config"], data["state"]
    if config["width"] % config["heads"] or config["context"] < 8:
        raise ValueError("Invalid checkpoint")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        f.write(b"FRG1")
        f.write(struct.pack("<4I", *(config[k] for k in ("width", "layers", "heads", "context"))))
        f.write(struct.pack("<I", len(state)))
        for name, tensor in state.items():
            tensor = tensor.detach().contiguous().float().cpu()
            name_bytes = name.encode("ascii")
            f.write(struct.pack("<H", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<B", tensor.ndim))
            for d in tensor.shape:
                f.write(struct.pack("<I", d))
            f.write(tensor.numpy().astype("<f4", copy=False).tobytes())
    print(f"Exported {len(state)} tensors to {output} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    export(args.checkpoint, args.output)
