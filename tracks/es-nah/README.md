# Track 1: Mexican Spanish - Nahuatl (Bilingual Code-Switching)
## Pipeline Overview
* **Base Model:** `openai/whisper-large-v3`
* **Fine-Tuning:** PEFT LoRA (Checkpoint-300 optimal)
* **Inference Strategy:**
  * Forced decoder language prompt: Spanish
  * Anti-hallucination settings: `condition_on_prev_tokens=False`, `repetition_penalty=1.2`, `no_repeat_ngram_size=3`
  * N-Best Beam Rescoring (`num_beams=5`) using pure Python 3-Gram LM (trained on training transcripts)
  * Post-processing: Regex lookup rules for agglutinative word boundaries (`tla-`, `que ma`, `y amiga`)
## Metric Progress

| Stage | WER | CER | Note |
| :--- | :--- | :--- | :--- |
| Zero-shot Baseline | 97.94% | - | High hallucination & language drift |
| Raw LoRA (ckpt-300) | 46.08% | 14.18% | Strong acoustics, word boundary issues |
| Custom Normalizer | 45.17% | 14.14% | Hand-crafted heuristic limits |
| Final Pipeline (N-Best + LM + Lookup) | **39.81%** | **12.11%** | Sub-40% WER achieved |

## Key Findings & Failure Analysis
1. **Acoustic vs Morphological Gap:** Low CER (12.11%) proves phonetic alignment is strong, but agglutinative morphology causes high WER due to word boundary segmentation penalties.
2. **Code-Switching Resiliency:** WER on code-switched/Spanish segments (38.84%) outperformed pure Nahuatl segments (40.46%), leveraging Whisper's Spanish pre-training prior.
3. **Generalization Caveat (Crucial):** Dataset audit reveals **100% speaker overlap** (5/5 speakers in both train & test splits) with minimal text leakage (4.69%). Local test WER reflects a **Speaker-Seen baseline**; performance on unseen speakers in hidden test set may vary.
