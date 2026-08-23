"""
tracks/es-en/train_lora.py
Training pipeline for fine-tuning Whisper Large-v3 on Miami Bangor (Spanish-English Code-Switching) using PEFT LoRA.
"""
import os
import glob
import re
import gc
import random
import torch
import torchaudio
from pydub import AudioSegment
from torch.utils.data import Dataset as TorchDataset
from dataclasses import dataclass
from typing import Any, Dict, List
from transformers import (
    WhisperForConditionalGeneration, 
    WhisperProcessor, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments
)
from peft import LoraConfig, get_peft_model
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

normalizer = BasicTextNormalizer()

def clean_transcript(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'&\=[a-zA-Z]+|@[a-z:&]+|\+[\.\/\<\>\?\!]*|xxx|yyy|www|[\[\]\(\)\<\>\_]', '', text)
    text = normalizer(text)
    text = re.sub(r'(didn|don|doesn|can|won|couldn|shouldn|wasn|weren|isn|aren)\s+t', r't', text)
    text = re.sub(r'(i|you|we|they|he|she|it)\s+(m|re|ve|ll|d|s)', r'', text)
    text = re.sub(r'(\w+)(?:\s+)+', r'', text)
    return ' '.join(text.split()).strip()

class MiamiDataset(TorchDataset):
    def __init__(self, data_list, processor):
        self.data = data_list
        self.processor = processor
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        wav, sr = torchaudio.load(item["audio_path"])
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        feat = self.processor.feature_extractor(wav.squeeze().numpy(), sampling_rate=16000).input_features[0]
        lbl = self.processor.tokenizer(item["sentence"]).input_ids
        return {"input_features": feat, "labels": lbl}

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        in_feats = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(in_feats, return_tensors="pt")
        lbl_feats = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(lbl_feats, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

def main():
    MODEL_ID = "openai/whisper-large-v3"
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    
    base_model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    base_model.config.forced_decoder_ids = None
    base_model.config.suppress_tokens = []
    base_model.config.use_cache = False

    peft_config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        target_modules=["q_proj", "v_proj"], 
        lora_dropout=0.05, 
        bias="none"
    )
    model = get_peft_model(base_model, peft_config)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./checkpoints/es-en-run",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        warmup_steps=15,
        max_steps=150,
        fp16=True,
        gradient_checkpointing=True,
        logging_steps=15,
        save_strategy="steps",
        save_steps=150,
        save_total_limit=1,
        report_to=["none"]
    )
    print("Trainer siap dijalankan dengan dataset Miami Bangor.")

if __name__ == "__main__":
    main()
