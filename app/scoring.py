import math
from typing import Dict, Any, List, Tuple

import numpy as np


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    return float(np.dot(vec_a, vec_b))


def topk_mean(values: List[float], k: int) -> float:
    if not values:
        return -1.0
    k = max(1, min(k, len(values)))
    return float(sum(sorted(values, reverse=True)[:k]) / k)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def get_cfg_for_classifier(calib: Dict[str, Any], key: str) -> Dict[str, float]:
    defaults = calib["global"]["decision_defaults"]
    override = calib.get("per_classifier_overrides", {}).get(key, {})
    return {
        "pass_score_threshold": override.get(
            "pass_score_threshold", defaults["pass_score_threshold"]
        ),
        "fail_score_threshold": override.get(
            "fail_score_threshold", defaults["fail_score_threshold"]
        ),
        "min_margin_for_pass": override.get(
            "min_margin_for_pass", defaults["min_margin_for_pass"]
        ),
        "min_separation_for_pass": override.get(
            "min_separation_for_pass", defaults["min_separation_for_pass"]
        ),
    }


def score_single_image(
    image_embedding: np.ndarray,
    classifier_set: Dict[str, Any],
    prompt_embeddings: Dict[str, Dict[str, List[Tuple[str, np.ndarray]]]],
    calibration: Dict[str, Any],
    return_top_k: int = 5,
) -> Dict[str, Any]:
    k_pos = calibration["global"]["top_k_positive"]
    k_neg = calibration["global"]["top_k_negative"]
    w_margin = calibration["global"]["weights"]["margin"]
    w_sep = calibration["global"]["weights"]["separation"]

    sigmoid_cfg = calibration["global"]["sigmoid"]
    use_sigmoid = sigmoid_cfg.get("enabled", True)
    a = sigmoid_cfg.get("a", 8.0)
    b = sigmoid_cfg.get("b", -0.2)

    temp = []
    for clf in classifier_set["classifiers"]:
        key = clf["key"]

        pos_scored = []
        for txt, emb in prompt_embeddings[key]["positive"]:
            pos_scored.append({"text": txt, "sim": cosine_similarity(image_embedding, emb)})

        neg_scored = []
        for txt, emb in prompt_embeddings[key]["negative"]:
            neg_scored.append({"text": txt, "sim": cosine_similarity(image_embedding, emb)})

        pos_scored.sort(key=lambda x: x["sim"], reverse=True)
        neg_scored.sort(key=lambda x: x["sim"], reverse=True)

        pos_topk_mean = topk_mean([x["sim"] for x in pos_scored], k_pos)
        neg_topk_mean = topk_mean([x["sim"] for x in neg_scored], k_neg)
        margin = pos_topk_mean - neg_topk_mean

        temp.append(
            {
                "key": key,
                "display_name": clf["display_name"],
                "pos_scored": pos_scored,
                "neg_scored": neg_scored,
                "pos_topk_mean": pos_topk_mean,
                "neg_topk_mean": neg_topk_mean,
                "margin": margin,
            }
        )

    pos_map = {t["key"]: t["pos_topk_mean"] for t in temp}
    results = []

    for item in temp:
        key = item["key"]
        competitors = sorted(
            [(k, v) for k, v in pos_map.items() if k != key],
            key=lambda x: x[1],
            reverse=True,
        )

        if competitors:
            top_comp_val = competitors[0][1]
            separation = item["pos_topk_mean"] - top_comp_val
        else:
            separation = item["pos_topk_mean"]

        raw_score = (w_margin * item["margin"]) + (w_sep * separation)
        score = sigmoid(a * raw_score + b) if use_sigmoid else raw_score

        cfg = get_cfg_for_classifier(calibration, key)
        pass_th = cfg["pass_score_threshold"]
        fail_th = cfg["fail_score_threshold"]
        min_margin = cfg["min_margin_for_pass"]
        min_sep = cfg["min_separation_for_pass"]

        if score >= pass_th and item["margin"] >= min_margin and separation >= min_sep:
            decision = "pass"
        elif score <= fail_th:
            decision = "fail"
        else:
            decision = "ambiguous"

        reason_codes = []
        if item["pos_topk_mean"] < 0.24:
            reason_codes.append("low_positive_signal")
        if item["neg_topk_mean"] > 0.22:
            reason_codes.append("high_negative_signal")
        if item["margin"] < min_margin:
            reason_codes.append("low_margin")
        if separation < min_sep:
            reason_codes.append("cross_classifier_conflict")
        if fail_th < score < pass_th:
            reason_codes.append("ambiguous_band")

        results.append(
            {
                "classifier_key": key,
                "display_name": item["display_name"],
                "decision": decision,
                "metrics": {
                    "pos_topk_mean": item["pos_topk_mean"],
                    "neg_topk_mean": item["neg_topk_mean"],
                    "margin": item["margin"],
                    "separation": separation,
                    "raw_score": raw_score,
                    "score": score,
                },
                "top_positive_matches": item["pos_scored"][:return_top_k],
                "top_negative_matches": item["neg_scored"][:return_top_k],
                "confusion_signals": [
                    {"classifier_key": ck, "competitor_pos_topk_mean": cv}
                    for ck, cv in competitors[:3]
                ],
                "reason_codes": reason_codes,
            }
        )

    pass_count = sum(1 for r in results if r["decision"] == "pass")
    fail_count = sum(1 for r in results if r["decision"] == "fail")
    amb_count = sum(1 for r in results if r["decision"] == "ambiguous")

    return {
        "overall_status": "all_pass" if pass_count == len(results) else "needs_review",
        "summary": {
            "total": len(results),
            "pass": pass_count,
            "fail": fail_count,
            "ambiguous": amb_count,
        },
        "results": results,
    }
