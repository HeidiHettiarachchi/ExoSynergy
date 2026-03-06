from dataclasses import dataclass
from typing import Dict, List, Optional
import math


# ---------------------------------------------------
# DATA STRUCTURES
# ---------------------------------------------------

@dataclass
class BiosignatureResult:
    name: str
    detected: bool
    reason: str
    gases_involved: List[str]


@dataclass
class AtmosphereSimilarity:
    planet: str
    similarity: float


@dataclass
class AtmosphericProfile:
    planet_type: str
    dominant_gas_fingerprint: str
    greenhouse_intensity_label: str
    toxicity_index: float
    toxicity_label: str
    atmosphere_similarity: List[AtmosphereSimilarity]


@dataclass
class HabitabilityResult:
    score: float
    grade: str
    category: str
    biosignatures: List[BiosignatureResult]
    factor_scores: Dict[str, float]
    summary: str
    profile: AtmosphericProfile


# ---------------------------------------------------
# MAIN DETECTOR
# ---------------------------------------------------

class BiosignatureDetector:

    # Solar system reference atmospheres (all values are fractions 0.0–1.0)
    SOLAR_SYSTEM = {
        "Earth":       {"N2": 0.7808, "O2": 0.2095, "CO2": 0.000415,
                        "H2O": 0.010,  "AR": 0.0093,  "CH4": 0.0000018},
        "Venus":       {"CO2": 0.965,  "N2": 0.035,   "SO2": 0.00015,
                        "H2O": 0.00002},
        "Mars":        {"CO2": 0.953,  "N2": 0.027,   "AR": 0.016,
                        "O2":  0.0013, "CO": 0.0008},
        "Jupiter":     {"H2":  0.896,  "HE": 0.102,   "CH4": 0.0003,
                        "NH3": 0.000026},
        "Saturn":      {"H2":  0.963,  "HE": 0.032,   "CH4": 0.0045,
                        "NH3": 0.000125},
        "Titan":       {"N2":  0.9484, "CH4": 0.0514, "H2":  0.001},
        "Uranus":      {"H2":  0.830,  "HE": 0.150,   "CH4": 0.023},
        "Neptune":     {"H2":  0.800,  "HE": 0.190,   "CH4": 0.015},
        "Early Earth": {"N2":  0.750,  "CO2": 0.120,  "CH4": 0.010,
                        "H2O": 0.030,  "H2":  0.002,  "NH3": 0.00005},
    }

    # Weighted scoring — reflects biological importance of each factor
    FACTOR_WEIGHTS = {
        "oxygen":             0.30,   # most critical — metabolic requirement
        "water":              0.25,   # single most agreed-upon prerequisite
        "greenhouse_penalty": 0.20,   # runaway CO2 is lethal (Venus/Mars)
        "low_toxicity":       0.15,   # SO2/H2S/CO kill life directly
        "nitrogen_buffer":    0.10,   # stabilises pressure, dilutes reactives
    }
    # temperature and size are optional — only included when actually known
    OPTIONAL_WEIGHTS = {
        "temperature": 0.15,
        "size":        0.10,
    }

    # Biosignature bonus weights (total capped at 20 pts)
    BIOSIG_WEIGHTS = {
        "Oxygen-Methane Disequilibrium": 15,
        "Ozone Shield":                  10,
        "Nitrous Oxide":                 10,
        "Ammonia Signature":              5,
        "Water Vapor":                    3,
    }

    # Score caps per atmosphere class
    SCORE_CAP = {
        "giant":       10.0,   # H2+He > 0.85 — true gas giant, no surface
        "sub_neptune": 35.0,   # H2+He 0.60–0.85 — possible but unlikely
        "rocky":       100.0,  # H2+He < 0.60 — full scoring
    }

    # ---------------------------------------------------
    # PUBLIC ENTRY POINT
    # ---------------------------------------------------

    def detect(
        self,
        gases: Dict[str, float],
        planet_temp_k: Optional[float] = None,
        planet_radius_rj: Optional[float] = None,
    ) -> HabitabilityResult:

        if not gases:
            raise ValueError("gas_predictions cannot be empty.")

        g = self._normalize({k.strip().upper(): float(v) for k, v in gases.items()})

        # Classify atmosphere type — used consistently everywhere
        h2_he = g.get("H2", 0) + g.get("HE", 0)
        if h2_he > 0.85:
            atm_class = "giant"
        elif h2_he > 0.60:
            atm_class = "sub_neptune"
        else:
            atm_class = "rocky"

        biosigs  = self._detect_biosignatures(g, atm_class)
        factors  = self._score_factors(g, atm_class, planet_temp_k, planet_radius_rj)
        score    = self._compute_score(factors, biosigs, atm_class, planet_temp_k, planet_radius_rj)
        grade    = self._grade(score)
        category = self._category(score)
        profile  = self._build_profile(g)
        summary  = self._summary(score, biosigs, factors)

        return HabitabilityResult(
            score=round(score, 1),
            grade=grade,
            category=category,
            biosignatures=biosigs,
            factor_scores={k: round(v, 3) for k, v in factors.items()},
            summary=summary,
            profile=profile,
        )

    # ---------------------------------------------------
    # NORMALIZE — clamp negatives, normalise to fractions
    # ---------------------------------------------------

    def _normalize(self, g: Dict[str, float]) -> Dict[str, float]:
        g = {k: max(0.0, v) for k, v in g.items()}
        total = sum(g.values())
        return g if total == 0 else {k: v / total for k, v in g.items()}

    # ---------------------------------------------------
    # BIOSIGNATURE DETECTION
    # atm_class: "giant" | "sub_neptune" | "rocky"
    # ---------------------------------------------------

    def _detect_biosignatures(self, g: Dict[str, float], atm_class: str) -> List[BiosignatureResult]:

        # True gas giants — no surface, detection not applicable
        if atm_class == "giant":
            return [BiosignatureResult(
                name="No Significant Biosignatures",
                detected=False,
                reason=(
                    "Gas giant atmosphere — biosignature detection not applicable. "
                    "No surface or liquid water context."
                ),
                gases_involved=[],
            )]

        o2  = g.get("O2",  0)
        ch4 = g.get("CH4", 0)
        o3  = g.get("O3",  0)
        n2o = g.get("N2O", 0)
        nh3 = g.get("NH3", 0)
        h2o = g.get("H2O", 0)

        # Sub-Neptunes — run detection but append uncertainty caveat to every result
        caveat = (
            " (Detected on H₂-rich envelope — habitability context uncertain, "
            "no confirmed rocky surface.)"
            if atm_class == "sub_neptune" else ""
        )

        results = []

        if o2 > 0.005 and ch4 > 0.0001:
            results.append(BiosignatureResult(
                name="Oxygen-Methane Disequilibrium",
                detected=True,
                reason=(
                    f"O₂ ({o2*100:.3f}%) and CH₄ ({ch4*100:.4f}%) coexist in "
                    f"atmospheric disequilibrium — simultaneous presence requires "
                    f"constant biological replenishment.{caveat}"
                ),
                gases_involved=["O2", "CH4"],
            ))

        if o3 > 0.000001:
            results.append(BiosignatureResult(
                name="Ozone Shield",
                detected=True,
                reason=(
                    f"O₃ ({o3*100:.6f}%) indicates sustained photochemical O₂ "
                    f"conversion and UV shielding — consistent with an oxygen-producing biosphere.{caveat}"
                ),
                gases_involved=["O3", "O2"],
            ))

        if n2o > 0.000001:
            results.append(BiosignatureResult(
                name="Nitrous Oxide",
                detected=True,
                reason=(
                    f"N₂O ({n2o*100:.5f}%) has no significant abiotic source at "
                    f"detectable concentrations — strong marker of biological denitrification.{caveat}"
                ),
                gases_involved=["N2O"],
            ))

        if nh3 > 0.00001:
            results.append(BiosignatureResult(
                name="Ammonia Signature",
                detected=True,
                reason=(
                    f"NH₃ ({nh3*100:.5f}%) is photochemically unstable — detectable "
                    f"levels suggest active biological nitrogen fixation.{caveat}"
                ),
                gases_involved=["NH3"],
            ))

        if h2o > 0.0001:
            results.append(BiosignatureResult(
                name="Water Vapor",
                detected=True,
                reason=(
                    f"H₂O ({h2o*100:.4f}%) — liquid water prerequisite detected. "
                    f"Necessary but not sufficient alone.{caveat}"
                ),
                gases_involved=["H2O"],
            ))

        if not results:
            results.append(BiosignatureResult(
                name="No Significant Biosignatures",
                detected=False,
                reason="No atmospheric gas combinations strongly associated with biological activity detected.",
                gases_involved=[],
            ))

        return results

    # ---------------------------------------------------
    # HABITABILITY FACTORS  (0.0 – 1.0 each)
    # ---------------------------------------------------

    def _score_factors(
        self,
        g: Dict[str, float],
        atm_class: str,
        temp_k: Optional[float],
        radius_rj: Optional[float],
    ) -> Dict[str, float]:

        factors: Dict[str, float] = {}

        if atm_class == "giant":
            # Informational only — score is hard-capped separately
            factors["water"]              = min(1.0, g.get("H2O", 0) * 20)
            factors["oxygen"]             = min(1.0, g.get("O2",  0) * 5)
            factors["nitrogen_buffer"]    = min(1.0, g.get("N2",  0) * 2)
            factors["greenhouse_penalty"] = 1.0
            factors["low_toxicity"]       = 1.0
            return factors

        h2o = g.get("H2O", 0)
        o2  = g.get("O2",  0)
        n2  = g.get("N2",  0)
        co  = g.get("CO",  0)
        so2 = g.get("SO2", 0)
        h2s = g.get("H2S", 0)
        co2 = g.get("CO2", 0)

        factors["water"]              = min(1.0, h2o * 20)
        factors["oxygen"]             = min(1.0, o2  * 5)
        factors["nitrogen_buffer"]    = min(1.0, n2  * 2)
        factors["greenhouse_penalty"] = max(0.0, 1.0 - co2 * 5)

        co_p  = min(1.0, co  * 50)
        so2_p = min(1.0, so2 * 100)
        h2s_p = min(1.0, h2s * 200)
        factors["low_toxicity"] = max(0.0, 1.0 - (co_p + so2_p + h2s_p) / 3)

        # Temperature — only when actually known (explicit None check so 0K works)
        if temp_k is not None:
            if 260 <= temp_k <= 320:
                factors["temperature"] = 1.0
            elif 220 <= temp_k <= 380:
                factors["temperature"] = 0.6
            else:
                factors["temperature"] = 0.2

        # Planet size — only when actually known
        if radius_rj is not None:
            if radius_rj < 0.2:
                factors["size"] = 1.0
            elif radius_rj < 0.5:
                factors["size"] = 0.8
            elif radius_rj < 1.5:
                factors["size"] = 0.4
            else:
                factors["size"] = 0.1

        return factors

    # ---------------------------------------------------
    # WEIGHTED SCORE
    # ---------------------------------------------------

    def _compute_score(
        self,
        factors: Dict[str, float],
        biosigs: List[BiosignatureResult],
        atm_class: str,
        temp_k: Optional[float],
        radius_rj: Optional[float],
    ) -> float:

        # Hard cap for true gas giants
        if atm_class == "giant":
            return self.SCORE_CAP["giant"]

        # Viability gate — both O2 and water essentially absent → life impossible
        water_f  = factors.get("water",  0)
        oxygen_f = factors.get("oxygen", 0)
        viability_capped = water_f < 0.05 and oxygen_f < 0.05

        # Build weight map from core weights
        weights = dict(self.FACTOR_WEIGHTS)

        # Add optional weights only when data is actually available
        if temp_k is not None:
            weights["temperature"] = self.OPTIONAL_WEIGHTS["temperature"]
        if radius_rj is not None:
            weights["size"] = self.OPTIONAL_WEIGHTS["size"]

        # Normalise so weights always sum to 1.0
        total_w = sum(weights.values())
        weights = {k: v / total_w for k, v in weights.items()}

        # Weighted base score
        base = sum(factors.get(k, 0) * w for k, w in weights.items()) * 100

        # Biosig bonus — weighted by confidence, capped at 20 pts
        bonus = min(
            sum(self.BIOSIG_WEIGHTS.get(b.name, 5) for b in biosigs if b.detected),
            20,
        )

        score = min(base + bonus, 100.0)

        # Apply viability gate
        if viability_capped:
            score = min(score, 30.0)

        # Apply sub-Neptune cap — uncertain habitability context
        if atm_class == "sub_neptune":
            score = min(score, self.SCORE_CAP["sub_neptune"])

        return score

    # ---------------------------------------------------
    # GRADE & CATEGORY
    # ---------------------------------------------------

    def _grade(self, score: float) -> str:
        if score >= 80: return "A"
        if score >= 60: return "B"
        if score >= 40: return "C"
        if score >= 20: return "D"
        return "E"

    def _category(self, score: float) -> str:
        if score >= 80: return "Highly Habitable"
        if score >= 60: return "Potentially Habitable"
        if score >= 40: return "Marginally Habitable"
        if score >= 20: return "Unlikely Habitable"
        return "Extremely Hostile"

    # ---------------------------------------------------
    # ATMOSPHERIC PROFILE
    # ---------------------------------------------------

    def _build_profile(self, g: Dict[str, float]) -> AtmosphericProfile:
        tox = self._toxicity_index(g)
        return AtmosphericProfile(
            planet_type               = self._planet_type(g),
            dominant_gas_fingerprint  = self._fingerprint(g),
            greenhouse_intensity_label= self._greenhouse_label(g),
            toxicity_index            = round(tox, 4),
            toxicity_label            = self._toxicity_label(tox),
            atmosphere_similarity     = self._atmosphere_similarity(g),
        )

    def _planet_type(self, g: Dict[str, float]) -> str:
        h2  = g.get("H2",  0)
        he  = g.get("HE",  0)
        co2 = g.get("CO2", 0)
        n2  = g.get("N2",  0)
        so2 = g.get("SO2", 0)
        o2  = g.get("O2",  0)

        # Thresholds match atm_class thresholds exactly — no inconsistency
        if h2 + he > 0.85: return "Gas Giant"
        if h2 + he > 0.60: return "Ice Giant / Sub-Neptune"
        if co2 > 0.70 and so2 > 0.0001: return "Venus-like (CO₂ + SO₂)"
        if co2 > 0.70 and n2  < 0.05:   return "Mars-like (CO₂-dominated)"
        if n2  > 0.40 and o2  > 0.10:   return "Earth-like"
        return "Rocky / Mixed Atmosphere"

    def _fingerprint(self, g: Dict[str, float]) -> str:
        top = sorted(g.items(), key=lambda x: x[1], reverse=True)[:3]
        return ", ".join(f"{k} {v*100:.1f}%" for k, v in top)

    def _greenhouse_intensity(self, g: Dict[str, float]) -> float:
        return min(1.0,
            g.get("CO2", 0)
            + g.get("CH4", 0) * 25
            + g.get("H2O", 0) * 0.5
            + g.get("N2O", 0) * 270
        )

    def _greenhouse_label(self, g: Dict[str, float]) -> str:
        v = self._greenhouse_intensity(g)
        if v < 0.05: return "Low"
        if v < 0.20: return "Moderate"
        if v < 0.50: return "High"
        return "Extreme"

    def _toxicity_index(self, g: Dict[str, float]) -> float:
        return min(1.0,
            g.get("CO",  0) * 50
            + g.get("SO2", 0) * 100
            + g.get("H2S", 0) * 200
        )

    def _toxicity_label(self, val: float) -> str:
        if val < 0.10: return "Low"
        if val < 0.30: return "Moderate"
        if val < 0.60: return "High"
        return "Extreme"

    # ---------------------------------------------------
    # ATMOSPHERE SIMILARITY  (all 9 bodies returned)
    # ---------------------------------------------------

    def _cosine(self, a: Dict[str, float], b: Dict[str, float]) -> float:
        keys = set(a) | set(b)
        dot  = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
        mag1 = math.sqrt(sum(a.get(k, 0) ** 2 for k in keys))
        mag2 = math.sqrt(sum(b.get(k, 0) ** 2 for k in keys))
        return 0.0 if mag1 == 0 or mag2 == 0 else dot / (mag1 * mag2)

    def _atmosphere_similarity(self, g: Dict[str, float]) -> List[AtmosphereSimilarity]:
        sims = [
            AtmosphereSimilarity(planet=p, similarity=round(self._cosine(g, ref) * 100, 1))
            for p, ref in self.SOLAR_SYSTEM.items()
        ]
        sims.sort(key=lambda x: x.similarity, reverse=True)
        return sims  # all 9 returned — frontend decides how many to show

    # ---------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------

    def _summary(self, score: float, biosigs: List[BiosignatureResult], factors: Dict[str, float]) -> str:
        detected = [b.name for b in biosigs if b.detected]
        limiting = [k for k, v in factors.items() if v < 0.4]
        return (
            f"Habitability score: {score:.1f}/100. "
            f"Biosignatures detected: {', '.join(detected) if detected else 'None'}. "
            f"Limiting factors: {', '.join(limiting) if limiting else 'None'}."
        )


# ---------------------------------------------------
# PUBLIC FUNCTION
# ---------------------------------------------------

def analyze_planet(
    gas_predictions: Dict[str, float],
    planet_temp_k: Optional[float] = None,
    planet_radius_rj: Optional[float] = None,
) -> HabitabilityResult:
    return BiosignatureDetector().detect(gas_predictions, planet_temp_k, planet_radius_rj)