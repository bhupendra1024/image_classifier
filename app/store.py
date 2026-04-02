import json
from pathlib import Path
from typing import Dict, Any

BASE_DIR = Path(__file__).resolve().parent.parent
CLASSIFIER_DIR = BASE_DIR / "configs" / "classifier_sets"
CALIB_DIR = BASE_DIR / "configs" / "calibration_profiles"


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_classifier_set(name: str, version: str) -> Dict[str, Any]:
    path = CLASSIFIER_DIR / f"{name}.json"
    data = _load_json(path)
    if data.get("version") != version:
        raise ValueError(
            f"classifier_set version mismatch: requested={version}, found={data.get('version')}"
        )
    return data


def load_calibration_profile(name: str, version: str) -> Dict[str, Any]:
    path = CALIB_DIR / f"{name}.json"
    data = _load_json(path)
    if data.get("version") != version:
        raise ValueError(
            f"calibration version mismatch: requested={version}, found={data.get('version')}"
        )
    return data
