import numpy as np
from transformers import WhisperProcessor
import evaluate

wer_metric = evaluate.load("wer")

def make_compute_metrics(processor: WhisperProcessor):
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids.copy()

        # Mask -100 sebelum decode (ini yang nge-fix nilai WER biar gak inflated)
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        # Normalize whitespace biar gak ada false-positive WER dari spasi ganda
        pred_str = [p.strip() for p in pred_str]
        label_str = [l.strip() for l in label_str]

        wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}
    return compute_metrics
