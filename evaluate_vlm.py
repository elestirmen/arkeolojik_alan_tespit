import base64
import json
import logging
import io
from pathlib import Path
from tqdm import tqdm
import argparse
from PIL import Image
import re

import vlm_detect
import vlm_lmstudio_detector
from vlm_lmstudio_detector import _parse_model_json

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger(__name__)

DATA_DIR = Path(r"C:\d_surucusu\arkeoloji\datalar\Val_norm_extracted\Val_norm")
POSITIVE_DIR = DATA_DIR / "Positive"
NEGATIVE_DIR = DATA_DIR / "Negative"
PROMPT_STAGE1_PATH = Path(r"c:\d_surucusu\arkeoloji\arkeolojik_alan_tespit\prompts\cappadocia_stage1.txt")
PROMPT_STAGE2_PATH = Path(r"c:\d_surucusu\arkeoloji\arkeolojik_alan_tespit\prompts\cappadocia_stage2_review.txt")

with open(PROMPT_STAGE1_PATH, "r", encoding="utf-8") as f:
    PROMPT_STAGE1_TEXT = f.read()

with open(PROMPT_STAGE2_PATH, "r", encoding="utf-8") as f:
    PROMPT_STAGE2_TEXT = f.read()

SCHEMA_STAGE1 = """{
  "candidates": [
    {
      "confidence": 0.0,
      "candidate_type": "mound | tumulus | ring_ditch | wall_trace | road_trace | foundation | enclosure | terrace | unknown",
      "bbox_xyxy": [0, 0, 0, 0],
      "visual_evidence": "...",
      "possible_false_positive": "...",
      "recommended_check": "rgb | hillshade | slrm | svf | ndsm | dsm | dtm | slope | field_check"
    }
  ],
  "visual_evidence": "short reason when candidates is empty",
  "possible_false_positive": "strongest non-archaeological explanation when candidates is empty",
  "recommended_check": "rgb | hillshade | slrm | svf | ndsm | dsm | dtm | slope | field_check"
}"""

SCHEMA_STAGE2 = """{
  "confirmed": true,
  "review_confidence": 0.0,
  "review_reason": "...",
  "review_false_positive": "...",
  "recommended_check": "rgb | hillshade | slrm | svf | ndsm | dsm | dtm | slope | field_check"
}"""

FINAL_PROMPT_STAGE1 = f"""Task: analyze this PNG image tile for possible archaeological features.
Data mode: RGB orthophoto only.
Decision guidance:
{PROMPT_STAGE1_TEXT}

Return exactly one JSON object with this schema and no markdown:
{SCHEMA_STAGE1}"""

FINAL_PROMPT_STAGE2 = f"""Task: second-pass archaeological review of the cropped feature.
Data mode: RGB orthophoto only.
Decision guidance:
{PROMPT_STAGE2_TEXT}

Return exactly one JSON object with this schema and no markdown:
{SCHEMA_STAGE2}"""

def encode_image(image_path: Path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def crop_and_encode_image(image_path: Path, bbox):
    try:
        img = Image.open(image_path)
        margin = 64
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(img.width, x2 + margin)
        y2 = min(img.height, y2 + margin)
        
        cropped = img.crop((x1, y1, x2, y2))
        buf = io.BytesIO()
        cropped.save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        LOGGER.error(f"Error cropping {image_path.name}: {e}")
        return encode_image(image_path)

def safe_model_name(model_str: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_]', '_', model_str).strip('_')

def evaluate():
    parser = vlm_detect.build_arg_parser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples per class")
    args, unknown = parser.parse_known_args()
    config = vlm_detect.build_config_from_args(args)
    limit = args.limit
    
    # Auto-start backend if needed
    try:
        vlm_detect._ensure_backend_ready(config)
    except Exception as e:
        LOGGER.warning(f"Backend hazir degil veya baslatilamadi: {e}")
    
    vlm_config = vlm_lmstudio_detector.VlmLmStudioConfig(
        base_url=config.base_url,
        api_key=config.api_key,
        model=config.model,
        temperature=config.temperature,
    )
    client = vlm_lmstudio_detector._make_openai_client(vlm_config)
    try:
        model = vlm_lmstudio_detector._resolve_lmstudio_model(client, vlm_config, logger=LOGGER)
    except Exception as e:
        LOGGER.error(f"Cannot connect to VLM Backend: {e}")
        return
    
    LOGGER.info(f"Using model: {model} on {config.base_url}")
    out_filename = f"vlm_evaluation_results_{safe_model_name(model)}.json"
    LOGGER.info(f"Results will be saved to: {out_filename}")

    positive_images = list(POSITIVE_DIR.rglob("*.png"))
    negative_images = list(NEGATIVE_DIR.rglob("*.png"))
    
    if limit is not None:
        positive_images = positive_images[:limit]
        negative_images = negative_images[:limit]
    
    LOGGER.info(f"Found {len(positive_images)} positive samples and {len(negative_images)} negative samples.")
    
    results = []

    def process_image(img_path, is_positive):
        base64_img = encode_image(img_path)
        # STAGE 1
        messages_stage1 = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": FINAL_PROMPT_STAGE1},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_img}"
                        }
                    }
                ]
            }
        ]
        
        try:
            res1 = client.chat.completions.create(
                model=model,
                messages=messages_stage1,
                temperature=vlm_config.temperature,
                max_tokens=2048
            )
            raw_text1 = res1.choices[0].message.content
            parsed1 = _parse_model_json(raw_text1)
            candidates = parsed1.get("candidates", [])
            if not isinstance(candidates, list):
                candidates = []
                
            best_candidate = None
            for c in candidates:
                if isinstance(c, dict) and c.get("confidence", 0) >= config.confidence_threshold:
                    if best_candidate is None or c.get("confidence", 0) > best_candidate.get("confidence", 0):
                        best_candidate = c
            
            # STAGE 2
            confirmed = False
            raw_text2 = None
            if best_candidate is not None:
                bbox = best_candidate.get("bbox_xyxy")
                if bbox and len(bbox) == 4:
                    base64_crop = crop_and_encode_image(img_path, bbox)
                else:
                    base64_crop = base64_img # fallback
                
                messages_stage2 = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": FINAL_PROMPT_STAGE2},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_crop}"
                                }
                            }
                        ]
                    }
                ]
                res2 = client.chat.completions.create(
                    model=model,
                    messages=messages_stage2,
                    temperature=vlm_config.temperature,
                    max_tokens=2048
                )
                raw_text2 = res2.choices[0].message.content
                parsed2 = _parse_model_json(raw_text2)
                confirmed = bool(parsed2.get("confirmed", False))
                # Update candidate confidence to review confidence if present
                rev_conf = parsed2.get("review_confidence")
                if rev_conf is not None and isinstance(rev_conf, (int, float)):
                    if rev_conf < config.confidence_threshold:
                        confirmed = False
            
            has_candidate_s1 = best_candidate is not None
            
            pred_stage1 = "positive" if has_candidate_s1 else "negative"
            pred_stage2 = "positive" if confirmed else "negative"
            
            return {
                "file": img_path.name,
                "label": "positive" if is_positive else "negative",
                "pred_stage1": pred_stage1,
                "pred_stage2": pred_stage2,
                "pred": pred_stage2, # fallback for backward compatibility
                "candidates_found": len(candidates),
                "raw_json_stage1": raw_text1,
                "raw_json_stage2": raw_text2,
                "confirmed_in_stage2": confirmed
            }
        except Exception as e:
            LOGGER.error(f"Error processing {img_path.name}: {e}")
            return None

    for img_path in tqdm(positive_images, desc="Evaluating Positive"):
        res = process_image(img_path, is_positive=True)
        if res: results.append(res)
        
    for img_path in tqdm(negative_images, desc="Evaluating Negative"):
        res = process_image(img_path, is_positive=False)
        if res: results.append(res)
        
    if not results:
        LOGGER.info("No results to evaluate.")
        return

    def calc_metrics(stage_key):
        tp = sum(1 for r in results if r['label'] == 'positive' and r.get(stage_key, r['pred']) == 'positive')
        fn = sum(1 for r in results if r['label'] == 'positive' and r.get(stage_key, r['pred']) == 'negative')
        tn = sum(1 for r in results if r['label'] == 'negative' and r.get(stage_key, r['pred']) == 'negative')
        fp = sum(1 for r in results if r['label'] == 'negative' and r.get(stage_key, r['pred']) == 'positive')
        total_pos = tp + fn
        total_neg = tn + fp
        accuracy = (tp + tn) / len(results) if results else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / total_pos if total_pos > 0 else 0
        return tp, fn, tn, fp, total_pos, total_neg, accuracy, precision, recall

    tp1, fn1, tn1, fp1, tp_tot, tn_tot, acc1, prec1, rec1 = calc_metrics('pred_stage1')
    tp2, fn2, tn2, fp2, _, _, acc2, prec2, rec2 = calc_metrics('pred_stage2')

    LOGGER.info("\n--- EVALUATION RESULTS (STAGE 1) ---")
    LOGGER.info(f"True Positives (TP): {tp1} / {tp_tot}")
    LOGGER.info(f"False Negatives (FN): {fn1} / {tp_tot}")
    LOGGER.info(f"True Negatives (TN): {tn1} / {tn_tot}")
    LOGGER.info(f"False Positives (FP): {fp1} / {tn_tot}")
    LOGGER.info(f"Accuracy:  {acc1:.2%} | Precision: {prec1:.2%} | Recall: {rec1:.2%}")

    LOGGER.info("\n--- EVALUATION RESULTS (STAGE 2) ---")
    LOGGER.info(f"True Positives (TP): {tp2} / {tp_tot}")
    LOGGER.info(f"False Negatives (FN): {fn2} / {tp_tot}")
    LOGGER.info(f"True Negatives (TN): {tn2} / {tn_tot}")
    LOGGER.info(f"False Positives (FP): {fp2} / {tn_tot}")
    LOGGER.info(f"Accuracy:  {acc2:.2%} | Precision: {prec2:.2%} | Recall: {rec2:.2%}")

    with open(out_filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    LOGGER.info(f"Saved detailed results to {out_filename}")

if __name__ == "__main__":
    evaluate()
