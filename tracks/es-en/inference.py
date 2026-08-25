"""
tracks/es-en/inference.py
Inference & Evaluation pipeline for Spanish-English Code-Switching ASR.
"""
import os
import re
import torch
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

class WhisperBilingualPipeline:
    def __init__(self, model_id="openai/whisper-large-v3", adapter_path=None, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.normalizer = BasicTextNormalizer()
        
        base_model = WhisperForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        
        if adapter_path and os.path.exists(adapter_path):
            self.model = PeftModel.from_pretrained(base_model, adapter_path)
            print(f"Loaded LoRA Adapter from: {adapter_path}")
        else:
            self.model = base_model
            
        self.model.eval()

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str): return ""
        text = re.sub(r'&\=[a-zA-Z]+|@[a-z:&]+|\+[\.\/\<\>\?\!]*|xxx|yyy|www|[\[\]\(\)\<\>\_]', '', text)
        text = self.normalizer(text)
        text = re.sub(r'(didn|don|doesn|can|won|couldn|shouldn|wasn|weren|isn|aren)\s+t', r't', text)
        text = re.sub(r'(i|you|we|they|he|she|it)\s+(m|re|ve|ll|d|s)', r'', text)
        text = re.sub(r'(\w+)(?:\s+)+', r'', text)
        return ' '.join(text.split()).strip()

    def transcribe(self, audio_path: str, language: str = None) -> str:
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
            
        inputs = self.processor.feature_extractor(
            waveform.squeeze().numpy(), 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_features.to(self.device).to(torch.float16 if self.device == "cuda" else torch.float32)
        
        gen_kwargs = {
            "max_length": 128, 
            "task": "transcribe", 
            "num_beams": 1, 
            "do_sample": False,
            "condition_on_prev_tokens": False,
            "repetition_penalty": 1.2,
            "no_repeat_ngram_size": 3
        }
        if language:
            gen_kwargs["language"] = language
            
        with torch.no_grad():
            pred_ids = self.model.generate(inputs, **gen_kwargs)
            pred_text = self.processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
            
        return self.clean_text(pred_text)
