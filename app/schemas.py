from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict


class RunOptions(BaseModel):
    return_top_k_matches: int = Field(default=5, ge=1, le=20)
    include_prompt_text: bool = True


class RunRequest(BaseModel):
    image_uri: str
    model_id: str
    classifier_set_name: str
    classifier_set_version: str
    calibration_profile_name: str
    calibration_profile_version: str
    options: RunOptions = RunOptions()


class MatchItem(BaseModel):
    text: Optional[str] = None
    sim: float


class ConfusionItem(BaseModel):
    classifier_key: str
    competitor_pos_topk_mean: float


class ClassifierMetrics(BaseModel):
    pos_topk_mean: float
    neg_topk_mean: float
    margin: float
    separation: float
    raw_score: float
    score: float


class ClassifierResult(BaseModel):
    classifier_key: str
    display_name: str
    decision: Literal["pass", "fail", "ambiguous"]
    metrics: ClassifierMetrics
    top_positive_matches: List[MatchItem]
    top_negative_matches: List[MatchItem]
    confusion_signals: List[ConfusionItem]
    reason_codes: List[str]


class RunResponse(BaseModel):
    run_id: str
    created_at: str
    model: Dict[str, str]
    classifier_set: Dict[str, str]
    calibration_profile: Dict[str, str]
    overall_status: Literal["all_pass", "needs_review"]
    summary: Dict[str, int]
    results: List[ClassifierResult]
