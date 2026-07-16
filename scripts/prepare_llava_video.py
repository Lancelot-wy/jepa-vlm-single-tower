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
import collections
import glob
import json
import os
import random

MEDIA_TOKENS = ("<image>", "<video>", "<|video_pad|>", "<|image_pad|>", "<|vision_start|>", "<|vision_end|>")


def iter_records(root: str, subsets: list[str], exclude_patterns: list[str]):
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

    exclude_patterns = [p.lower() for p in exclude_patterns]
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
                if not vp:
                    continue
                # A basename-only benchmark overlap check cannot establish that
                # train/test splits from a common upstream collection are clean.
                # Keep enough provenance in the manifest to audit that decision,
                # and allow callers to exclude an entire upstream source here.
                provenance = f"{os.path.relpath(fp, root)}\n{vp}".lower()
                if any(p in provenance for p in exclude_patterns):
                    continue
                yield resolve_video_path(vp), d, os.path.relpath(fp, root)


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
    subset_group = ap.add_mutually_exclusive_group()
    subset_group.add_argument("--subsets", nargs="+", default=None,
                              help="subset name substrings to include (match jsonl filenames)")
    subset_group.add_argument("--all-subsets", action="store_true",
                              help="use every matching jsonl below --root")
    ap.add_argument("--exclude-patterns", nargs="*", default=[],
                    help="case-insensitive substrings matched against source-jsonl and video path")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-videos", type=int, default=0, help="0 = no cap")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--qa", action="store_true", help="also write qa_train.jsonl (SFT baseline / Phase B)")
    ap.add_argument("--qa-per-video", type=int, default=2)
    ap.add_argument("--no-check-files", action="store_true",
                    help="skip os.path.exists filtering (faster, but may keep missing files)")
    args = ap.parse_args()

    # Do reservoir sampling over the *whole* eligible copy.  Stopping after the
    # first N paths biases the mix towards alphabetically early source files.
    # The reservoir keeps memory bounded by --max-videos while remaining exactly
    # uniform over the discovered unique videos.
    subsets = [] if args.all_subsets else (args.subsets or ["0_30_s_academic_v0_1"])
    rng = random.Random(args.seed)
    seen_paths: set[str] = set()
    selected: list[tuple[str, dict, str]] = []
    n_missing = 0
    n_eligible = 0
    for vp, rec, source_jsonl in iter_records(args.root, subsets, args.exclude_patterns):
        if vp in seen_paths:
            continue
        if not args.no_check_files and not os.path.exists(vp):
            n_missing += 1
            continue
        seen_paths.add(vp)
        n_eligible += 1
        item = (vp, rec, source_jsonl)
        if not args.max_videos or len(selected) < args.max_videos:
            selected.append(item)
        else:
            slot = rng.randrange(n_eligible)
            if slot < args.max_videos:
                selected[slot] = item

    rng.shuffle(selected)
    n_val = max(1, int(len(selected) * args.val_frac)) if selected else 0
    val, train = selected[:n_val], selected[n_val:]

    os.makedirs(args.out_dir, exist_ok=True)
    for name, rows in [("train.jsonl", train), ("val.jsonl", val)]:
        with open(os.path.join(args.out_dir, name), "w") as out:
            for vp, _, source_jsonl in rows:
                out.write(json.dumps({"video": vp, "source_jsonl": source_jsonl}) + "\n")
    print(f"eligible videos: {n_eligible} (missing {n_missing}) -> selected {len(selected)} "
          f"-> train {len(train)} / val {len(val)}")
    if args.max_videos and n_eligible > args.max_videos:
        print(f"reservoir sample: {args.max_videos}/{n_eligible}, seed={args.seed}")
    selected_sources = collections.Counter(source for _, _, source in selected)
    print("selected source inventory:")
    for source, count in selected_sources.most_common():
        print(f"  {count:7d}  {source}")

    if args.qa:
        n_qa = 0
        qa_path = os.path.join(args.out_dir, "qa_train.jsonl")
        with open(qa_path, "w") as out:
            for vp, rec, source_jsonl in train:
                for q, a in extract_qa(rec, args.qa_per_video):
                    out.write(json.dumps({"video": vp, "question": q, "answer": a,
                                          "source_jsonl": source_jsonl},
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
