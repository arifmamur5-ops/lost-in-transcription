"""
common/inference.py

Shared inference logic for all 3 tracks. Loads a fine-tuned Whisper model
and runs transcription on a single audio file or a batch.

NOTE: Official runtime repository specification from DrivenData has not been
released yet (as of this writing). This script assumes a reasonable generic
contract (load model -> read audio -> output transcript) that should be easy
to adapt once the real spec drops. Do not treat file paths / CLI args below
as final until cross-checked against the spec.

Usage (standalone test):
    python inference.py --model_dir tracks/es-nah/model --audio_path sample.wav

Usage (as a module, e.g. from main.py):
    from common.inference import load_model, transcribe
"""

import argparse
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import soundfile as sf
import librosa


def load_model(model_dir: str, device: str = None):
    """
    Loads a fine-tuned Whisper model + processor from a local directory.
    model_dir should contain the saved model (pytorch_model.bin / safetensors,
    config.json) and processor files (tokenizer, feature_extractor configs),
    i.e. the output of model.save_pretrained() + processor.save_pretrained().
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = torch.float16 if device == "cuda" else torch.float32

    model = WhisperForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype=dtype
    ).to(device)
    model.eval()

    processor = WhisperProcessor.from_pretrained(model_dir)

    return model, processor, device


def load_audio(audio_path: str, target_sr: int = 16000):
    """
    Loads an audio file and resamples to 16kHz mono if needed.
    Uses librosa for robustness across formats (wav, mp3, flac, etc).
    """
    audio_array, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    return audio_array, target_sr


def transcribe(
    model,
    processor,
    device: str,
    audio_array,
    sampling_rate: int,
    language: str = None,
    task: str = "transcribe",
    max_new_tokens: int = 225,
    num_beams: int = 1,
) -> str:
    """
    Runs inference on a single audio array, returns transcript string.

    language: forced language hint (e.g. "es"). Leave None to let the model
    auto-detect — for code-switched tracks, auto-detect performed better in
    zero-shot baselines (see notebooks/zeroshot_baseline results).
    """
    inputs = processor(
        audio_array, sampling_rate=sampling_rate, return_tensors="pt"
    ).to(device)

    dtype = torch.float16 if device == "cuda" else torch.float32
    inputs["input_features"] = inputs["input_features"].to(dtype)

    forced_decoder_ids = None
    if language:
        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=language, task=task
        )

    with torch.no_grad():
        pred_ids = model.generate(
            inputs["input_features"],
            forced_decoder_ids=forced_decoder_ids,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )

    transcript = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
    return transcript.strip()


def transcribe_file(model_dir: str, audio_path: str, language: str = None) -> str:
    """Convenience wrapper: load model + audio + run inference in one call.
    Useful for quick manual testing, not for batch/container inference
    (which should load the model once and reuse it across files)."""
    model, processor, device = load_model(model_dir)
    audio_array, sr = load_audio(audio_path)
    return transcribe(model, processor, device, audio_array, sr, language=language)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Whisper inference on a single audio file.")
    parser.add_argument("--model_dir", required=True, help="Path to fine-tuned model directory")
    parser.add_argument("--audio_path", required=True, help="Path to audio file to transcribe")
    parser.add_argument("--language", default=None, help="Optional forced language hint, e.g. 'es'")
    args = parser.parse_args()

    result = transcribe_file(args.model_dir, args.audio_path, language=args.language)
    print(f"Transcript: {result}")
