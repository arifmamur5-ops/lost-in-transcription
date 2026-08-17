"""
common/data_prep.py

Shared data loading & preprocessing pipeline for all 3 tracks:
- Spanish-English (es-en)
- Spanish-Nahuatl (es-nah)
- Indonesian-Javanese (id-jv)

Usage:
    from common.data_prep import load_track_data, prepare_dataset

    dataset = load_track_data(
        audio_dir="tracks/es-nah/data/audio",
        transcript_file="tracks/es-nah/data/transcripts.tsv",
    )
    dataset = dataset.map(lambda batch: prepare_dataset(batch, processor))
"""

import os
import pandas as pd
from datasets import Dataset, DatasetDict, Audio


def load_track_data(
    audio_dir: str,
    transcript_file: str,
    sampling_rate: int = 16000,
    test_size: float = 0.1,
    seed: int = 42,
    audio_col: str = "audio_file",
    transcript_col: str = "transcript",
    sep: str = "\t",
    min_duration: float = 0.5,
    max_duration: float = 30.0,
) -> DatasetDict:
    """
    Loads a transcript file + matching audio files into a HF DatasetDict
    with train/test split, ready for feature extraction.

    Assumes transcript_file has at minimum two columns: audio filename and transcript text.
    Adjust `audio_col` / `transcript_col` / `sep` to match actual MDC file format
    once confirmed (this is a reasonable default assumption based on common TSV formats).
    """
    df = pd.read_csv(transcript_file, sep=sep)

    # sanity check on expected columns
    if audio_col not in df.columns or transcript_col not in df.columns:
        raise ValueError(
            f"Expected columns '{audio_col}' and '{transcript_col}' not found. "
            f"Found columns: {list(df.columns)}. Adjust audio_col/transcript_col args."
        )

    # drop rows with missing transcript or audio path
    df = df.dropna(subset=[audio_col, transcript_col]).reset_index(drop=True)

    # build full audio path
    df[audio_col] = df[audio_col].apply(lambda x: os.path.join(audio_dir, x))

    # filter out rows where audio file doesn't actually exist
    df["_exists"] = df[audio_col].apply(os.path.exists)
    missing_count = (~df["_exists"]).sum()
    if missing_count > 0:
        print(f"[data_prep] Warning: {missing_count} audio files not found, dropping those rows.")
    df = df[df["_exists"]].drop(columns="_exists").reset_index(drop=True)

    # normalize transcript whitespace
    df[transcript_col] = df[transcript_col].astype(str).str.strip()
    df = df[df[transcript_col].str.len() > 0].reset_index(drop=True)

    # rename to standard column names used downstream
    df = df.rename(columns={audio_col: "audio_file", transcript_col: "transcript"})

    ds = Dataset.from_pandas(df[["audio_file", "transcript"]])
    ds = ds.cast_column("audio_file", Audio(sampling_rate=sampling_rate))

    # filter by duration (removes invalid/too-short/too-long clips)
    def _valid_duration(example):
        duration = len(example["audio_file"]["array"]) / example["audio_file"]["sampling_rate"]
        return min_duration <= duration <= max_duration

    ds = ds.filter(_valid_duration)

    split = ds.train_test_split(test_size=test_size, seed=seed)
    return split


def prepare_dataset(batch, processor):
    """
    Maps a raw batch (audio_file, transcript) into Whisper-ready
    input_features and labels. Use with .map(), not .map(batched=True)
    unless you vectorize this further.
    """
    audio = batch["audio_file"]

    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    batch["labels"] = processor.tokenizer(batch["transcript"]).input_ids

    return batch


def get_dataset_stats(dataset: DatasetDict) -> dict:
    """Quick sanity-check stats after loading — run this before training."""
    stats = {}
    for split_name, split_data in dataset.items():
        durations = [
            len(ex["audio_file"]["array"]) / ex["audio_file"]["sampling_rate"]
            for ex in split_data
        ]
        stats[split_name] = {
            "num_examples": len(split_data),
            "total_hours": sum(durations) / 3600,
            "avg_duration_sec": sum(durations) / len(durations) if durations else 0,
            "min_duration_sec": min(durations) if durations else 0,
            "max_duration_sec": max(durations) if durations else 0,
        }
    return stats


if __name__ == "__main__":
    # quick smoke test — adjust paths before running
    ds = load_track_data(
        audio_dir="tracks/es-nah/data/audio",
        transcript_file="tracks/es-nah/data/transcripts.tsv",
    )
    print(ds)
    print(get_dataset_stats(ds))
