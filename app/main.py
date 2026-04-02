from datetime import datetime, timezone
import uuid

from fastapi import FastAPI, HTTPException

from app.schemas import RunRequest, RunResponse
from app.store import load_classifier_set, load_calibration_profile
from app.embedding import embed_image, load_prompt_embeddings
from app.scoring import score_single_image

app = FastAPI(title="Visual QA Compliance API", version="1.0.0")


@app.get("/health")
def health() -> dict:
    return {"ok": True, "service": "visual-qa-compliance-api"}


@app.post("/v1/runs", response_model=RunResponse)
def run_visual_qa(req: RunRequest):
    try:
        classifier_set = load_classifier_set(
            req.classifier_set_name, req.classifier_set_version
        )
        calibration = load_calibration_profile(
            req.calibration_profile_name, req.calibration_profile_version
        )

        prompt_embeddings = load_prompt_embeddings(req.model_id, classifier_set)
        image_embedding = embed_image(req.model_id, req.image_uri)

        scored = score_single_image(
            image_embedding=image_embedding,
            classifier_set=classifier_set,
            prompt_embeddings=prompt_embeddings,
            calibration=calibration,
            return_top_k=req.options.return_top_k_matches,
        )

        if not req.options.include_prompt_text:
            for result in scored["results"]:
                for match in result["top_positive_matches"]:
                    match.pop("text", None)
                for match in result["top_negative_matches"]:
                    match.pop("text", None)

        return {
            "run_id": f"run_{datetime.now(timezone.utc).strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": {"model_id": req.model_id},
            "classifier_set": {
                "name": req.classifier_set_name,
                "version": req.classifier_set_version,
            },
            "calibration_profile": {
                "name": req.calibration_profile_name,
                "version": req.calibration_profile_version,
            },
            "overall_status": scored["overall_status"],
            "summary": scored["summary"],
            "results": scored["results"],
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"run_failed: {str(exc)}") from exc
