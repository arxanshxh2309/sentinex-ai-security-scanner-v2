"""
Sentinex AI - Bayesian threat estimation.

The API uses a fast closed-form Beta posterior by default so scans return
reliably on CPU. Offline experiments can still enable PyMC sampling by passing
``n_samples > 0``.
"""

from typing import Dict, List, Tuple

import numpy as np


# Threat priors per weapon class (higher = more dangerous)
CLASS_THREAT_PRIORS = {
    "handgun": {"alpha": 9.0, "beta": 1.0},
    "dagger": {"alpha": 8.0, "beta": 2.0},
    "sword": {"alpha": 7.0, "beta": 2.5},
    "knife": {"alpha": 6.0, "beta": 3.0},
    "scissors": {"alpha": 3.0, "beta": 5.0},
}

THREAT_LEVELS = {
    "CRITICAL": (0.85, 1.0),
    "HIGH": (0.65, 0.85),
    "MEDIUM": (0.40, 0.65),
    "LOW": (0.15, 0.40),
    "CLEAR": (0.00, 0.15),
}


def estimate_threat(
    detections: List[Dict],
    n_samples: int = 0,
) -> Dict:
    """
    Estimate threat probability from YOLO detections.

    Args:
        detections: Dicts with keys such as class, confidence, bbox_area_ratio.
        n_samples: MCMC samples to draw. Use 0 for the production fast path.

    Returns:
        Dict with threat_probability, threat_level, uncertainty, and details.
    """
    if not detections:
        return {
            "threat_probability": 0.0,
            "threat_level": "CLEAR",
            "uncertainty": 0.0,
            "confidence_interval": [0.0, 0.0],
            "detections": [],
            "message": "No threats detected - bag is clear.",
        }

    object_threats = []

    for det in detections:
        cls = str(det["class"]).lower()
        conf = float(det.get("confidence", 0.5))
        area_ratio = float(det.get("bbox_area_ratio", 0.05))
        mean_threat, ci_low, ci_high = _estimate_object_threat(
            cls=cls,
            confidence=conf,
            area_ratio=area_ratio,
            n_samples=n_samples,
        )

        object_threats.append(
            {
                "class": cls,
                "confidence": conf,
                "threat_score": mean_threat,
                "ci_95": [ci_low, ci_high],
            }
        )

    # Aggregate by worst-case threat, which is appropriate for screening.
    overall_threat = max(t["threat_score"] for t in object_threats)
    overall_ci_low = min(t["ci_95"][0] for t in object_threats)
    overall_ci_high = max(t["ci_95"][1] for t in object_threats)
    uncertainty = overall_ci_high - overall_ci_low

    threat_level = "CLEAR"
    for level, (low, high) in THREAT_LEVELS.items():
        if low <= overall_threat < high:
            threat_level = level
            break
    if overall_threat >= 0.85:
        threat_level = "CRITICAL"

    return {
        "threat_probability": round(overall_threat, 4),
        "threat_level": threat_level,
        "uncertainty": round(uncertainty, 4),
        "confidence_interval": [round(overall_ci_low, 4), round(overall_ci_high, 4)],
        "detections": object_threats,
        "message": f"Threat level: {threat_level} ({overall_threat:.1%} probability)",
    }


def _weighted_observation(confidence: float, area_ratio: float) -> float:
    confidence = float(np.clip(confidence, 0.0, 1.0))
    area_score = float(np.clip(area_ratio * 5.0, 0.0, 1.0))
    return confidence * 0.7 + area_score * 0.3


def _estimate_object_threat(
    cls: str,
    confidence: float,
    area_ratio: float,
    n_samples: int,
) -> Tuple[float, float, float]:
    prior = CLASS_THREAT_PRIORS.get(cls, {"alpha": 3.0, "beta": 5.0})
    weighted_obs = _weighted_observation(confidence, area_ratio)

    if n_samples > 0:
        return _estimate_with_mcmc(prior, weighted_obs, n_samples)

    # Conjugate approximation: stronger detections count as more evidence.
    evidence_strength = 2.0 + confidence * 8.0 + min(area_ratio * 10.0, 1.0) * 4.0
    alpha = prior["alpha"] + weighted_obs * evidence_strength
    beta = prior["beta"] + (1.0 - weighted_obs) * evidence_strength
    mean = alpha / (alpha + beta)
    variance = (alpha * beta) / (((alpha + beta) ** 2) * (alpha + beta + 1.0))
    margin = 1.96 * float(np.sqrt(variance))
    ci_low = float(np.clip(mean - margin, 0.0, 1.0))
    ci_high = float(np.clip(mean + margin, 0.0, 1.0))
    return float(mean), ci_low, ci_high


def _estimate_with_mcmc(
    prior: Dict[str, float],
    weighted_obs: float,
    n_samples: int,
) -> Tuple[float, float, float]:
    import pymc as pm

    with pm.Model():
        threat = pm.Beta("threat", alpha=prior["alpha"], beta=prior["beta"])
        pm.Normal("obs", mu=threat, sigma=0.1, observed=weighted_obs)
        trace = pm.sample(
            draws=n_samples,
            tune=min(1000, max(250, n_samples // 2)),
            chains=2,
            cores=1,
            progressbar=False,
            random_seed=42,
            return_inferencedata=True,
        )

    posterior = trace.posterior["threat"].values.flatten()
    return (
        float(np.mean(posterior)),
        float(np.percentile(posterior, 2.5)),
        float(np.percentile(posterior, 97.5)),
    )


if __name__ == "__main__":
    sample_detections = [
        {"class": "knife", "confidence": 0.87, "bbox_area_ratio": 0.05},
        {"class": "handgun", "confidence": 0.92, "bbox_area_ratio": 0.12},
    ]

    result = estimate_threat(sample_detections, n_samples=1000)

    print("=" * 50)
    print("SENTINEX AI - THREAT ASSESSMENT")
    print("=" * 50)
    print(f"Overall Threat:  {result['threat_probability']:.1%}")
    print(f"Threat Level:    {result['threat_level']}")
    print(f"Uncertainty:     +/-{result['uncertainty']:.1%}")
    print(f"95% CI:          {result['confidence_interval']}")
    print("\nPer-object breakdown:")
    for det in result["detections"]:
        print(f"  {det['class']:12s} | threat: {det['threat_score']:.3f} | CI: {det['ci_95']}")
