import json
import glob
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger(__name__)

def generate_report():
    files = glob.glob("vlm_evaluation_results_*.json")
    if not files:
        LOGGER.info("No evaluation results found to compare.")
        return

    LOGGER.info("Generating Comparison Report...\n")
    
    md_table = "| Model (Stage) | Accuracy | Precision | Recall | F1 Score | TP | FN | TN | FP |\n"
    md_table += "|---------------|----------|-----------|--------|----------|----|----|----|----|\n"

    for file_path in files:
        model_name = Path(file_path).stem.replace("vlm_evaluation_results_", "")
        with open(file_path, "r", encoding="utf-8") as f:
            results = json.load(f)
            
        def get_row(stage_name, stage_key):
            tp = sum(1 for r in results if r['label'] == 'positive' and r.get(stage_key, r.get('pred', 'negative')) == 'positive')
            fn = sum(1 for r in results if r['label'] == 'positive' and r.get(stage_key, r.get('pred', 'negative')) == 'negative')
            tn = sum(1 for r in results if r['label'] == 'negative' and r.get(stage_key, r.get('pred', 'negative')) == 'negative')
            fp = sum(1 for r in results if r['label'] == 'negative' and r.get(stage_key, r.get('pred', 'negative')) == 'positive')
            
            total_pos = tp + fn
            total_neg = tn + fp
            
            accuracy = (tp + tn) / len(results) if results else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / total_pos if total_pos > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            return f"| {model_name} ({stage_name}) | {accuracy:.2%} | {precision:.2%} | {recall:.2%} | {f1:.2%} | {tp} | {fn} | {tn} | {fp} |\n"

        # Check if the result has stage1 keys. If so, print both. Else, print just Stage1.
        has_stage1 = 'pred_stage1' in results[0] if results else False
        
        if has_stage1:
            md_table += get_row("Stage 1", "pred_stage1")
            md_table += get_row("Stage 2", "pred_stage2")
        else:
            md_table += get_row("Stage 1", "pred") # Backwards compat

    print(md_table)
    
    with open("benchmark_report.md", "w", encoding="utf-8") as f:
        f.write("# VLM Models Benchmark Report\n\n")
        f.write(md_table)
        
    LOGGER.info("Report saved to benchmark_report.md")

if __name__ == "__main__":
    generate_report()
