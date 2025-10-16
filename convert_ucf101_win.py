"""
modified from https://huggingface.co/docs/transformers/tasks/video_classification
and https://huggingface.co/docs/transformers/main/en/model_doc/video_llava

Usage: "python convert_ucf101_win.py --tmp_dir .\tmp\ucf101 --out_dir .\converted_ucf101"

"""
import argparse
import json
import pathlib
import shutil
import tarfile
import time
import os
import stat

import av
import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image
from pathlib import Path


def read_video_pyav(container, indices):
    """Decode frames with PyAV."""
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def video_to_images(video_path, num_images=8):
    """
    Extract frames from a video file and return PIL images.
    IMPORTANT: Use context manager so the file handle is closed on Windows.
    """
    with av.open(video_path) as container:
        total_frames = container.streams.video[0].frames
        # guard: if frames unknown or 0, fallback
        if not total_frames or total_frames <= 0:
            total_frames = 32  # heuristic fallback
        indices = np.linspace(0, max(total_frames - 1, 0), num=min(num_images, total_frames)).astype(int)
        video_array = read_video_pyav(container, indices)
    images = [Image.fromarray(frame).convert("RGB") for frame in video_array]
    return images


def safe_rmtree(path: Path, retries: int = 6, delay: float = 0.5):
    """
    Robust rmtree for Windows:
      - retries with backoff if PermissionError (file in use / thumbnailer / indexer)
      - clears read-only attributes on failure
    """
    def _on_rm_error(func, p, exc_info):
        try:
            os.chmod(p, stat.S_IWRITE)
        except Exception:
            pass
        # try once more
        try:
            func(p)
        except Exception:
            pass

    for i in range(retries):
        try:
            shutil.rmtree(path, onerror=_on_rm_error)
            return
        except PermissionError:
            time.sleep(delay * (i + 1))
        except FileNotFoundError:
            return
    # last attempt
    try:
        shutil.rmtree(path, onerror=_on_rm_error)
    except Exception:
        # give up silently; or raise if好み
        pass


def main(tmp_dir, out_dir):
    hf_dataset_identifier = "sayakpaul/ucf101-subset"
    filename = "UCF101_subset.tar.gz"
    file_path = hf_hub_download(repo_id=hf_dataset_identifier, filename=filename, repo_type="dataset")

    tmp_path = Path(tmp_dir)
    tmp_path.mkdir(parents=True, exist_ok=True)
    with tarfile.open(file_path) as t:
        t.extractall(tmp_path)

    dataset_root_path = tmp_path / "UCF101_subset"
    all_video_file_paths = (
        list(dataset_root_path.glob("train/*/*.avi"))
        + list(dataset_root_path.glob("val/*/*.avi"))
        + list(dataset_root_path.glob("test/*/*.avi"))
    )

    # --- FIX 1: OS非依存でクラス名を取得 ---
    labels = set()
    for p in all_video_file_paths:
        parts = Path(p).parts
        if len(parts) >= 2:
            labels.add(parts[-2])
        else:
            print(f"[warn] path too short to get class: {p}")
    class_labels = sorted(labels)
    prompt = f'Classify the video into one of the following classes: {", ".join(class_labels)}.'

    # convert all videos
    split2examples = {"train": [], "val": [], "test": []}
    out_path = Path(out_dir)
    out_image_path = out_path / "images"

    for i, video_file_path in enumerate(all_video_file_paths):
        p = Path(video_file_path)
        # parts: .../<split>/<label>/<file.avi>
        if len(p.parts) < 3:
            print(f"[warn] unexpected path structure: {p}")
            continue
        split = p.parts[-3]
        label = p.parts[-2]

        images = video_to_images(video_file_path)

        image_path_prefix = "/".join(p.with_suffix("").parts[-3:])
        split2examples[split].append(
            {
                "id": f"{split}-{i:010d}",
                "source": "ucf101",
                "conversations": [
                    {
                        "images": [f"{image_path_prefix}.{k}.jpg" for k in range(len(images))],
                        "user": prompt,
                        "assistant": label,
                    }
                ],
            }
        )
        (out_image_path / image_path_prefix).parent.mkdir(parents=True, exist_ok=True)
        for k, image in enumerate(images):
            image.save((out_image_path / image_path_prefix).with_suffix(f".{k}.jpg"))

    out_path.mkdir(parents=True, exist_ok=True)
    for split, examples in split2examples.items():
        with open(out_path / f"ucf101_{split}.jsonl", "w", encoding="utf-8") as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")

    # --- FIX 2: Windowsでの削除を安全に ---
    safe_rmtree(tmp_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # --- FIX 3: Windowsでも安全な相対tmpを既定にする ---
    parser.add_argument("--tmp_dir", type=str, default=".tmp/ucf101")
    parser.add_argument("--out_dir", type=str, default="./ucf101")
    args = parser.parse_args()

    main(args.tmp_dir, args.out_dir)
