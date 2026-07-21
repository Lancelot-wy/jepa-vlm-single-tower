import json
import subprocess
import sys

import numpy as np
import torch

from jepa_vlm.modeling.state_loss import DistributedRunningCenter


def test_running_center_state_round_trip():
    center = DistributedRunningCenter(3, 0.99)
    target = torch.tensor([[[1.0, 2.0, 3.0]]])
    center.update(target, torch.ones(1, 1, dtype=torch.bool))
    state = center.state_dict()
    restored = DistributedRunningCenter(3, 0.99)
    restored.load_state_dict(state)
    assert restored.running_center.dtype == torch.float32
    assert torch.equal(restored.running_center, center.running_center)
    assert restored.updates.item() == 1


def _write_video(path):
    import av

    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=4)
        stream.width = 64
        stream.height = 64
        stream.pix_fmt = "yuv420p"
        for index in range(40):
            image = np.zeros((64, 64, 3), dtype=np.uint8)
            image[..., 0] = index * 5
            image[:, index % 64, 1] = 255
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_tiny_training_checkpoint_and_resume(tmp_path):
    video = tmp_path / "video.mp4"
    _write_video(video)
    manifest = tmp_path / "train.jsonl"
    with manifest.open("w") as handle:
        for index in range(2):
            handle.write(json.dumps({
                "video": str(video),
                "question": "What changes?",
                "answer": "motion",
                "source_dataset": "unit",
                "source_id": f"row-{index}",
            }) + "\n")
    output = tmp_path / "run"
    base = [
        sys.executable, "-m", "jepa_vlm.train",
        "--config", "configs/orca_token_sweep/a1_query_k4.yaml",
        "model.tiny_config=true", "model.pretrained=tiny", "model.dtype=float32",
        "model.frame_size=64", f"train.text_manifest={manifest}",
        f"train.output_dir={output}", "train.num_workers=0", "train.grad_accum=1",
        "train.temporal_qa_ratio=0.0",
        "train.gradient_checkpointing=false", "train.log_every=1", "train.save_every=1",
    ]
    subprocess.run(base + ["train.max_steps=1"], check=True, stdout=subprocess.DEVNULL)
    first = torch.load(output / "checkpoint-1/state.pt", map_location="cpu", weights_only=False)
    subprocess.run(
        base + ["train.max_steps=2", f"train.resume={output / 'checkpoint-1'}"],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    second = torch.load(output / "checkpoint-2/state.pt", map_location="cpu", weights_only=False)
    assert second["step"] == 2
    assert second["data_batches_seen"] == 2
    assert second["model_aux"]["state_center"]["updates"] > first["model_aux"]["state_center"]["updates"]
    meta = json.loads((output / "checkpoint-2/checkpoint_meta.json").read_text())
    assert meta["step"] == 2
    assert meta["state_bytes"] == (output / "checkpoint-2/state.pt").stat().st_size
