#!/usr/bin/env python3
"""Export fine-tuned Kiki SLM to GGUF for Ollama.

Run inside the Colab training notebook AFTER training finishes. Produces:
  1. Merged fp16 safetensors → /content/drive/MyDrive/kiki-slm/merged/
     (fallback — can be converted to GGUF offline if step 2 fails)
  2. Q4_K_M GGUF → /content/drive/MyDrive/kiki-slm/gguf/kiki-sft-v1-Q4_K_M.gguf
     (~2.6 GB — this is what Ollama loads)

Usage in Colab notebook:
    !python scripts/export_gguf.py \\
        --adapter-dir /content/drive/MyDrive/kiki-slm/adapters/kiki-sft-v1 \\
        --drive-out /content/drive/MyDrive/kiki-slm

Or paste as a cell. The script is idempotent — re-running skips steps whose
outputs already exist.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path


def _log(msg: str) -> None:
    print(f"── [GGUF EXPORT] {msg}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter-dir", required=True,
        help="Directory containing the trained LoRA adapter (DRIVE_OUT from notebook).",
    )
    parser.add_argument(
        "--drive-out", default="/content/drive/MyDrive/kiki-slm",
        help="Root of the Drive kiki-slm directory (where merged/ and gguf/ land).",
    )
    parser.add_argument(
        "--quantization", default="q4_k_m",
        help="llama.cpp quantization method (q4_k_m=2.6GB default, q5_k_m=3.1GB, q8_0=4.5GB).",
    )
    parser.add_argument(
        "--skip-merged-save", action="store_true",
        help="Don't save the fp16 merged checkpoint to Drive (saves ~8GB + 5min).",
    )
    args = parser.parse_args()

    adapter_dir = args.adapter_dir
    drive_root = Path(args.drive_out)
    merged_out_drive = drive_root / "merged" / "kiki-sft-v1"
    gguf_out_drive = drive_root / "gguf"
    gguf_out_drive.mkdir(parents=True, exist_ok=True)

    # Use the LATEST checkpoint-* inside adapter_dir if present, else adapter_dir itself
    import glob
    checkpoints = sorted(glob.glob(f"{adapter_dir}/checkpoint-*"))
    load_from = checkpoints[-1] if checkpoints else adapter_dir
    _log(f"loading adapter from: {load_from}")

    # ------------------------------------------------------------------
    # 1. Load fp16 (NOT 4-bit — merging requires full precision)
    # ------------------------------------------------------------------
    from unsloth import FastLanguageModel

    _log("loading base model + adapter (fp16, no 4-bit)...")
    t0 = time.perf_counter()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=load_from,
        max_seq_length=4096,
        load_in_4bit=False,      # critical: must be fp16 to merge
        dtype=None,
    )
    _log(f"loaded in {time.perf_counter() - t0:.1f}s")

    # ------------------------------------------------------------------
    # 2. Save merged fp16 to Drive (recovery checkpoint)
    # ------------------------------------------------------------------
    if not args.skip_merged_save:
        if merged_out_drive.exists() and any(merged_out_drive.iterdir()):
            _log(f"merged fp16 already exists at {merged_out_drive} — skipping")
        else:
            _log(f"saving merged fp16 → {merged_out_drive}")
            merged_out_drive.mkdir(parents=True, exist_ok=True)
            t0 = time.perf_counter()
            model.save_pretrained_merged(
                str(merged_out_drive),
                tokenizer,
                save_method="merged_16bit",
            )
            _log(f"merged fp16 saved in {time.perf_counter() - t0:.1f}s")

    # ------------------------------------------------------------------
    # 3. Export GGUF (Q4_K_M by default, ~2.6 GB)
    # ------------------------------------------------------------------
    # Note: Unsloth's save_pretrained_gguf APPENDS "_gguf" to whatever path
    # you give it. So passing "/content/kiki-gguf" actually writes to
    # "/content/kiki-gguf_gguf/". We hunt in both locations to be robust.
    local_gguf_dir = Path("/content/kiki-gguf")
    unsloth_actual_dir = Path("/content/kiki-gguf_gguf")
    for d in (local_gguf_dir, unsloth_actual_dir):
        if d.exists():
            shutil.rmtree(d)
    local_gguf_dir.mkdir(parents=True)

    _log(f"exporting GGUF (method={args.quantization})... this takes 5-15 min")
    t0 = time.perf_counter()
    try:
        model.save_pretrained_gguf(
            str(local_gguf_dir),
            tokenizer,
            quantization_method=args.quantization,
        )
    except Exception as e:
        _log(f"GGUF export failed: {e}")
        _log("FALLBACK: the merged fp16 is at " + str(merged_out_drive))
        _log("Download it to your Mac and run `llama.cpp/convert_hf_to_gguf.py` locally.")
        sys.exit(1)
    _log(f"GGUF exported in {time.perf_counter() - t0:.1f}s")

    # Find the .gguf file that Unsloth produced and copy to Drive.
    # Search BOTH the path we asked for and the "_gguf" suffix path Unsloth uses.
    candidate_dirs = [unsloth_actual_dir, local_gguf_dir]
    gguf_files: list[Path] = []
    found_in: Path | None = None
    for d in candidate_dirs:
        if d.exists():
            gguf_files = list(d.rglob("*.gguf"))
            if gguf_files:
                found_in = d
                break
    if not gguf_files:
        _log(f"no .gguf file found under {[str(d) for d in candidate_dirs]} — check Unsloth version")
        sys.exit(1)
    _log(f"located GGUF in {found_in}: {[f.name for f in gguf_files]}")

    for src in gguf_files:
        dst = gguf_out_drive / f"kiki-sft-v1-{args.quantization.upper()}.gguf"
        _log(f"copying {src.name} ({src.stat().st_size / 1024**3:.2f} GB) → {dst}")
        shutil.copy(src, dst)
        _log(f"✓ {dst}")

    # ------------------------------------------------------------------
    # 4. Modelfile — prefer the one Unsloth wrote (has the right TEMPLATE),
    #    fall back to a minimal one if Unsloth didn't generate it.
    # ------------------------------------------------------------------
    modelfile_path = gguf_out_drive / "Modelfile"
    unsloth_modelfile = (found_in / "Modelfile") if found_in else None
    if unsloth_modelfile and unsloth_modelfile.exists():
        # Rewrite the FROM line to point at the renamed GGUF in Drive
        original = unsloth_modelfile.read_text()
        rewritten_lines = []
        for line in original.splitlines():
            if line.strip().lower().startswith("from "):
                rewritten_lines.append(f"FROM ./kiki-sft-v1-{args.quantization.upper()}.gguf")
            else:
                rewritten_lines.append(line)
        modelfile_path.write_text("\n".join(rewritten_lines) + "\n")
        _log(f"wrote Modelfile (from Unsloth, FROM line rewritten) → {modelfile_path}")
    else:
        modelfile_content = f"""FROM ./kiki-sft-v1-{args.quantization.upper()}.gguf

PARAMETER temperature 0.1
PARAMETER num_ctx 4096
PARAMETER num_predict 1024
PARAMETER stop "<|im_end|>"
"""
        modelfile_path.write_text(modelfile_content)
        _log(f"wrote Modelfile (minimal fallback) → {modelfile_path}")

    _log("DONE. To load into Ollama locally:")
    _log(f"  1. Download {gguf_out_drive}/kiki-sft-v1-*.gguf + Modelfile from Drive")
    _log("  2. ollama create kiki-sft-v1 -f Modelfile")
    _log("  3. curl http://localhost:11434/api/chat -d '{...}'")


if __name__ == "__main__":
    main()
