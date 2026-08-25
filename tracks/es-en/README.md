# Track 2: Spanish - English (Bilingual Code-Switching)

## Pipeline Overview
* **Base Model:** `openai/whisper-large-v3`
* **Fine-Tuning:** PEFT LoRA (Rank `r=16`, `lora_alpha=32`, Target Modules: `q_proj`, `v_proj`)
* **Inference Strategy:**
  * Decoding configuration: Greedy search (`num_beams=1`), FP16 precision.
  * Anti-hallucination guards: `condition_on_prev_tokens=False`, `repetition_penalty=1.2`, `no_repeat_ngram_size=3`.
  * Task: Automatic code-switching transcription (`task="transcribe"`).
  * Post-Processing: Custom bilingual normalizer combining `BasicTextNormalizer`, contraction repair (`didn't`, `i'm`), and stutter-reduction regex.
* **Weights Storage:** Checkpoint adapter weights (`adapter_model.safetensors`) are hosted privately on Google Drive / Kaggle Datasets per competition rules.

## Metric Progress

| Stage | Subset / Split | WER | Note |
| :--- | :--- | :--- | :--- |
| Zero-shot Baseline | Miami Bangor Validation | 24.80% | Language confusion & foreign token hallucinations |
| Raw LoRA Adapter | Sub-sample (50 Utterances) | **9.84%** | Optimal acoustic alignment, zero repetition loops |
| Full Benchmark (300 Samples) | Code-Switching (Mix) | **13.25%** | Robust cross-lingual boundary transitions |
| Full Benchmark (300 Samples) | Pure English (Casual) | **16.75%** | Casual conversational phonetics & slang reductions |
| **Final Evaluated Pipeline** | **300 Samples Full Scale** | **13.01%** | **Sub-15% overall WER target achieved** |

## Key Findings & Failure Analysis
1. **Acoustic Reductions & Slang:** Error analysis indicates the majority of substitutions stem from conversational contractions (e.g., *"kind of"* vs *"kinda"*, *"going to"* vs *"gonna"*) and homophonic plosives rather than language drift.
2. **Code-Switching Boundary Stability:** The LoRA adapter successfully prevents decoder looping on intra-sentential language switches (Spanish <-> English), retaining bilingual word choices without hallucinatory script drift.
3. **Cross-Talk & Short Interjections:** Short utterances (<1.5s) containing background conversational overlap (*cross-talk* / laughing) account for minor token deletion penalties.

## Theoretical Baseline & References
This track's methodology and baseline considerations build upon the joint LID-ASR paradigm on spontaneous bilingual speech:
* **Hillah, L., Dubiel, M., & Leiva, L. A. (2024).** *"¿Te vienes? Sure!" Joint Fine-tuning of Language Detection and Transcription Improves Automatic Recognition of Code-Switching Speech.* In ACM Conversational User Interfaces (CUI '24). https://doi.org/10.1145/3640794.3665579
