#!/usr/bin/env python3
"""Export a Whisper model to OpenVINO IR format for GPU inference."""
import sys
from pathlib import Path

from optimum.intel import OVModelForSpeechSeq2Seq
from transformers import AutoProcessor

MODEL_MAP = {
    "tiny": "openai/whisper-tiny",
    "base": "openai/whisper-base",
    "small": "openai/whisper-small",
    "medium": "openai/whisper-medium",
}


def main():
    model_size = sys.argv[1] if len(sys.argv) > 1 else "small"

    if model_size not in MODEL_MAP:
        print("Usage: python3 export_whisper_openvino.py [tiny|base|small|medium]")
        sys.exit(1)

    model_id = MODEL_MAP[model_size]
    output_dir = f"models/whisper-{model_size}-openvino"

    if Path(output_dir).exists():
        print(f"[!] {output_dir} already exists. Delete it to re-export.")
        sys.exit(0)

    print(f"[→] Downloading and exporting {model_id} to OpenVINO format...")
    print(f"    This will take a few minutes and requires ~2GB download.")

    model = OVModelForSpeechSeq2Seq.from_pretrained(model_id, export=True)
    model.save_pretrained(output_dir)

    processor = AutoProcessor.from_pretrained(model_id)
    processor.save_pretrained(output_dir)

    print(f"[✓] Model exported to {output_dir}")
    print(f"    Set config.yaml stt.device to 'auto' or 'gpu' to use it.")


if __name__ == "__main__":
    main()
