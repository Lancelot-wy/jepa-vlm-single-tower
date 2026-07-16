"""Build jepa-vlm manifests from a LOCAL LLaVA-Video-178K copy.

LLaVA-Video ships per-subset jsonl under <root>/jsonl/final_<subset>_*_processed.jsonl
whose lines carry an absolute `video_path` to an already-extracted video. Phase A is
self-supervised (only `video` is needed): we dedup video paths, drop missing files,
and split into train/val manifests.

With --qa we additionally extract {video, question, answer} pairs from each record's
`conversations` (LLaVA format: alternating human/gpt turns) into qa_train.jsonl for
the SFT baseline / Phase B. QA pairs are taken ONLY from train-split videos so the
val manifest stays clean for probes. NOTE: verify the conversations schema against
one line of your local jsonl (`head -1 ... | python -m json.tool`); adjust
extract_qa() if the field names differ.

  python scripts/prepare_llava_video.py \
      --root /data/vjuicefs_ai_ocr_wl/public_data/video_data/LLaVA-Video-178K \
      --subsets 0_30_s_academic_v0_1 \
      --out-dir /data/vjuicefs_sz_ocr_wl/public_data/11193960/jepa_data/llava_video \
      --max-videos 8000 --qa

Then set in your config (paths are absolute -> data_root=""):
  train.train_manifest=<out>/train_flow.jsonl   (after scripts/compute_flow.py)
  train.val_manifest=<out>/val.jsonl  train.data_root=""
"""

import argparse
import glob
import json
import os
import random

MEDIA_TOKENS = ("<image>", "<video>", "<|video_pad|>", "<|image_pad|>", "<|vision_start|>", "<|vision_end|>")


def iter_records(root: str, subsets: list[str]):
    jsonl_dir = os.path.join(root, "jsonl")
    files = sorted(glob.glob(os.path.join(jsonl_dir, "final_*_processed*.jsonl")))
    # The shared copy is also distributed as one directory per subset rather
    # than a single <root>/jsonl directory. Support both layouts.
    if not files:
        files = sorted(glob.glob(
            os.path.join(root, "**", "final_*_processed*.jsonl"),
            recursive=True,
        ))
    if subsets:
        files = [f for f in files if any(
            s in os.path.basename(f) or s in os.path.relpath(f, root)
            for s in subsets
        )]
    if not files:
        raise FileNotFoundError(f"no matching jsonl in {jsonl_dir} for subsets={subsets}")

    def resolve_video_path(vp: str) -> str:
        """Resolve paths embedded in a source copy from another machine."""
        if os.path.exists(vp):
            return vp
        marker = "LLaVA-Video-178K"
        if marker in vp:
            suffix = vp.split(marker, 1)[1].lstrip(os.sep)
            candidate = os.path.join(root, suffix)
            if os.path.exists(candidate):
                return candidate
        if not os.path.isabs(vp):
            candidate = os.path.join(root, vp)
            if os.path.exists(candidate):
                return candidate
        return vp

    for fp in files:
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                vp = d.get("video_path") or d.get("video")
                if vp:
                    yield resolve_video_path(vp), d


def extract_qa(rec: dict, max_pairs: int = 2) -> list[tuple[str, str]]:
    """Tolerant QA extraction from LLaVA-style `conversations` (human/gpt turns)."""
    convs = rec.get("conversations") or rec.get("QA") or []
    pairs, q = [], None
    for turn in convs:
        who = (turn.get("from") or turn.get("role") or "").lower()
        text = (turn.get("value") or turn.get("content") or "").strip()
        for tok in MEDIA_TOKENS:
            text = text.replace(tok, "").strip()
        if who in ("human", "user"):
            q = text
        elif who in ("gpt", "assistant") and q:
            pairs.append((q, text))
            q = None
        if len(pairs) >= max_pairs:
            break
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="LLaVA-Video-178K root")
    ap.add_argument("--subsets", nargs="*", default=["0_30_s_academic_v0_1"],
                    help="subset name substrings to include (match jsonl filenames)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-videos", type=int, default=0, help="0 = no cap")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--qa", action="store_true", help="also write qa_train.jsonl (SFT baseline / Phase B)")
    ap.add_argument("--qa-per-video", type=int, default=2)
    ap.add_argument("--no-check-files", action="store_true",
                    help="skip os.path.exists filtering (faster, but may keep missing files)")
    args = ap.parse_args()

    seen: dict[str, dict] = {}
    n_missing = 0
    for vp, rec in iter_records(args.root, args.subsets):
        if vp in seen:
            continue
        if not args.no_check_files and not os.path.exists(vp):
            n_missing += 1
            continue
        seen[vp] = rec
        if args.max_videos and len(seen) >= args.max_videos:
            break

    vids = sorted(seen)
    random.Random(args.seed).shuffle(vids)
    n_val = max(1, int(len(vids) * args.val_frac)) if vids else 0
    val, train = vids[:n_val], vids[n_val:]

    os.makedirs(args.out_dir, exist_ok=True)
    for name, rows in [("train.jsonl", train), ("val.jsonl", val)]:
        with open(os.path.join(args.out_dir, name), "w") as out:
            for vp in rows:
                out.write(json.dumps({"video": vp}) + "\n")
    print(f"videos: {len(vids)} (missing {n_missing}) -> train {len(train)} / val {len(val)}")

    if args.qa:
        n_qa = 0
        qa_path = os.path.join(args.out_dir, "qa_train.jsonl")
        with open(qa_path, "w") as out:
            for vp in train:
                for q, a in extract_qa(seen[vp], args.qa_per_video):
                    out.write(json.dumps({"video": vp, "question": q, "answer": a},
                                         ensure_ascii=False) + "\n")
                    n_qa += 1
        print(f"qa pairs: {n_qa} -> {qa_path}")
        if n_qa == 0:
            print("WARNING: no QA extracted - the conversations schema likely differs; "
                  "inspect one source line and adjust extract_qa().")
        else:
            with open(qa_path) as f:
                print("sample:", f.readline().strip()[:200])

    print(f"wrote {args.out_dir}/train.jsonl , val.jsonl")
    print("set: train.data_root='' ; run scripts/compute_flow.py on train.jsonl next")


if __name__ == "__main__":
    main()
