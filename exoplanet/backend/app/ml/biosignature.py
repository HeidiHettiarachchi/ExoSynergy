from dataclasses import dataclass
from typing import Dict, List, Optional
import math

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
    greenhouse_heating_index: float
    atmospheric_density: str
    thermal_stability: str
    temperature_potential: str
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

class BiosignatureDetector:

    SOLAR_SYSTEM = {
        "Earth":       {"N2": 0.7808, "O2": 0.2095, "CO2": 0.000415, "H2O": 0.010,  "O3": 0.000001},
        "Venus":       {"CO2": 0.9650, "N2": 0.0350, "SO2": 0.00015},
        "Mars":        {"CO2": 0.9530, "N2": 0.0270, "O2": 0.0013,   "CO": 0.0007},
        "Jupiter":     {"H2": 0.8960,  "HE": 0.1020, "CH4": 0.0003},
        "Saturn":      {"H2": 0.9630,  "HE": 0.0320, "CH4": 0.0045},
        "Titan":       {"N2": 0.9484,  "CH4": 0.0514, "H2": 0.001},
        "Uranus":      {"H2": 0.8300,  "HE": 0.1500, "CH4": 0.0230},
        "Neptune":     {"H2": 0.8000,  "HE": 0.1900, "CH4": 0.0150},
        "Early Earth": {"N2": 0.7500,  "CO2": 0.1200, "CH4": 0.0100, "H2O": 0.020},
    }

    FACTOR_WEIGHTS = {
        "oxygen":             0.30,
        "water":              0.25,
        "greenhouse_penalty": 0.20,
        "low_toxicity":       0.15,
        "nitrogen_buffer":    0.10,
    }

    BIOSIG_WEIGHTS = {
        "Oxygen-Methane Disequilibrium": 15,
        "Ozone Shield":                  10,
        "Nitrous Oxide":                 10,
        "Ammonia Signature":              5,
        "Water Vapor":                    3,
    }

    SCORE_CAP = {
        "giant":       10.0,
        "sub_neptune": 35.0,
        "rocky":       100.0,
    }

    GAS_KEYS = [
        "H2O", "CO2", "CH4", "CO", "NH3",
        "H2",  "HE",  "N2",  "O2",  "O3",
        "SO2", "H2S", "N2O"
    ]

    _CO2_REF = 0.000280

    _TOXICITY_PARAMS = {
        "CO":  {"idlh": 0.001200, "weight": 0.40},  
        "SO2": {"idlh": 0.000100, "weight": 0.80}, 
        "H2S": {"idlh": 0.000300, "weight": 1.20},   
        "NH3": {"idlh": 0.003000, "weight": 0.20},  
    }

    def detect(self, gases: Dict[str, float],
        transmission_data: Optional[Dict[str, float]] = None) -> HabitabilityResult:
        
        g = self._normalize({k.strip().upper(): float(v) for k, v in gases.items()})

        h2_he = g.get("H2", 0) + g.get("HE", 0)

        if h2_he > 0.85:
            atm_class = "giant"
        elif h2_he > 0.60:
            atm_class = "sub_neptune"
        else:
            atm_class = "rocky"

        biosigs = self._detect_biosignatures(g, atm_class)
        factors = self._score_factors(g, atm_class)
        score   = self._compute_score(factors, biosigs, atm_class)

        return HabitabilityResult(
            score         = round(score, 1),
            grade         = self._grade(score),
            category      = self._category(score),
            biosignatures = biosigs,
            factor_scores = {k: round(v, 3) for k, v in factors.items()},
            summary       = self._summary(score, biosigs, factors),
            profile       = self._build_profile(g, transmission_data),
        )

    # ---------------------------------------------------
    # NORMALIZE
    # ---------------------------------------------------

    def _normalize(self, g: Dict[str, float]) -> Dict[str, float]:
        g = {k.strip().upper(): max(0.0, v) for k, v in g.items()}
        total = sum(g.values())
        normalized = g if total == 0 else {k: v / total for k, v in g.items()}
        return {k: normalized.get(k, 0.0) for k in self.GAS_KEYS}

    # ---------------------------------------------------
    # BIOSIGNATURE HELPERS
    # ---------------------------------------------------

    def _is_oxygen_methane_disequilibrium(self, g: Dict[str, float]) -> bool:
        return g.get("O2", 0) >= 0.01 and g.get("CH4", 0) >= 0.0001

    def _is_ozone_shield(self, g: Dict[str, float]) -> bool:
        return g.get("O3", 0) >= 1e-7 and g.get("O2", 0) >= 0.01

    def _is_n2o_indicator(self, g: Dict[str, float]) -> bool:
        return g.get("N2O", 0) >= 5e-7

    def _is_ammonia_signature(self, g: Dict[str, float]) -> bool:
        return g.get("NH3", 0) >= 5e-6 and g.get("H2", 0) < 0.05

    # ---------------------------------------------------
    # BIOSIGNATURE DETECTION
    # ---------------------------------------------------

    def _detect_biosignatures(self, g: Dict[str, float], atm_class: str) -> List[BiosignatureResult]:

        biosigs = []

        if atm_class == "giant":
            return biosigs

        if atm_class == "sub_neptune":
            if self._is_oxygen_methane_disequilibrium(g):
                biosigs.append(BiosignatureResult(
                    name           = "Oxygen-Methane Disequilibrium",
                    detected       = True,
                    reason         = (
                        f"O2 ({g.get('O2',0)*100:.2f}%) and CH4 ({g.get('CH4',0)*100:.3f}%) coexist in a disequilibrium state. "
                        "This can indicate ongoing chemical production."
                    ),
                    gases_involved = ["O2", "CH4"],
                ))

            if self._is_ozone_shield(g):
                biosigs.append(BiosignatureResult(
                    name           = "Ozone Shield",
                    detected       = True,
                    reason         = (
                        f"O3 at {g.get('O3',0)*1e6:.2f} ppm suggests an oxygen-rich upper atmosphere. "
                        "This is often associated with photosynthetic oxygen production."
                    ),
                    gases_involved = ["O3", "O2"],
                ))

            if self._is_n2o_indicator(g):
                biosigs.append(BiosignatureResult(
                    name           = "Nitrous Oxide",
                    detected       = True,
                    reason         = (
                        f"N2O at {g.get('N2O',0)*1e6:.2f} ppm may indicate biological nitrogen cycling."
                    ),
                    gases_involved = ["N2O"],
                ))

            if g.get("H2O", 0) > 0.01:
                biosigs.append(BiosignatureResult(
                    name           = "Water Vapor Indicator",
                    detected       = True,
                    reason         = (
                        f"H2O at {g.get('H2O',0)*100:.2f}% indicates a water-rich atmosphere. "
                        "This is important for habitability, though not a direct biosignature."
                    ),
                    gases_involved = ["H2O"],
                ))

            return biosigs

        if self._is_oxygen_methane_disequilibrium(g):
            biosigs.append(BiosignatureResult(
                name           = "Oxygen-Methane Disequilibrium",
                detected       = True,
                reason         = (
                    f"O2 ({g.get('O2',0)*100:.2f}%) and CH4 ({g.get('CH4',0)*100:.3f}%) coexist in a disequilibrium state. "
                    "This combination is difficult to sustain without active replenishment."
                ),
                gases_involved = ["O2", "CH4"],
            ))

        if self._is_ozone_shield(g):
            biosigs.append(BiosignatureResult(
                name           = "Ozone Shield",
                detected       = True,
                reason         = (
                    f"O3 at {g.get('O3',0)*1e6:.2f} ppm indicates significant oxygen photochemistry. "
                    "It is a secondary indicator of an oxygen-rich atmosphere."
                ),
                gases_involved = ["O3", "O2"],
            ))

        if self._is_n2o_indicator(g):
            biosigs.append(BiosignatureResult(
                name           = "Nitrous Oxide",
                detected       = True,
                reason         = (
                    f"N2O at {g.get('N2O',0)*1e6:.2f} ppm is a potential sign of biological nitrogen cycling."
                ),
                gases_involved = ["N2O"],
            ))

        if self._is_ammonia_signature(g):
            biosigs.append(BiosignatureResult(
                name           = "Ammonia Signature",
                detected       = True,
                reason         = (
                    f"NH3 at {g.get('NH3',0)*1e6:.2f} ppm with low H2 suggests a transient nitrogen chemistry. "
                    "This may be consistent with biological or volcanic sources."
                ),
                gases_involved = ["NH3", "H2"],
            ))

        if g.get("H2O", 0) > 0.01:
            biosigs.append(BiosignatureResult(
                name           = "Water Vapor Indicator",
                detected       = True,
                reason         = (
                    f"H2O at {g.get('H2O',0)*100:.2f}% indicates a moist atmosphere. "
                    "Liquid water is a key requirement for life as we know it."
                ),
                gases_involved = ["H2O"],
            ))

        return biosigs

    # ---------------------------------------------------
    # FACTOR SCORING  
    # ---------------------------------------------------

    def _score_factors(self, g: Dict[str, float], atm_class: str) -> Dict[str, float]:

        factors = {}

        o2 = g.get("O2", 0)
        if o2 <= 0.0:
            factors["oxygen"] = 0.0
        elif o2 < 0.05:
            factors["oxygen"] = (o2 / 0.05) * 0.3
        elif o2 <= 0.25:
            factors["oxygen"] = 0.3
        elif o2 <= 0.35:
            factors["oxygen"] = 0.2
        else:
            factors["oxygen"] = 0.1

        h2o = g.get("H2O", 0)
        if h2o <= 0.0:
            factors["water"] = 0.0
        elif h2o <= 0.03:
            factors["water"] = min(1.0, h2o / 0.03)
        else:
            factors["water"] = max(0.0, 1.0 - (h2o - 0.03) / 0.07)

        ghi = self._greenhouse_heating_index(g)
        if ghi < 0.15:
            factors["greenhouse_penalty"] = 0.15
        elif ghi <= 0.35:
            factors["greenhouse_penalty"] = 0.60
        elif ghi <= 0.55:
            factors["greenhouse_penalty"] = 1.0
        elif ghi <= 0.75:
            factors["greenhouse_penalty"] = 0.60
        else:
            factors["greenhouse_penalty"] = 0.10

        tox = self._toxicity_index(g)
        factors["low_toxicity"] = max(0.0, 1.0 - tox)

        n2 = g.get("N2", 0)
        if n2 <= 0.0:
            factors["nitrogen_buffer"] = 0.0
        elif n2 <= 0.78:
            factors["nitrogen_buffer"] = min(1.0, n2 / 0.78)
        else:
            factors["nitrogen_buffer"] = max(0.0, 1.0 - (n2 - 0.78) / 0.22)

        return factors

    # ---------------------------------------------------
    # FINAL SCORE
    # ---------------------------------------------------

    def _compute_score(self, factors: Dict[str, float],
                       biosigs: List[BiosignatureResult],
                       atm_class: str) -> float:

        score = sum(
            factors.get(name, 0) * weight * 100
            for name, weight in self.FACTOR_WEIGHTS.items()
        )

        for b in biosigs:
            if b.detected:
                score += self.BIOSIG_WEIGHTS.get(b.name, 0)

        if atm_class == "sub_neptune":
            score *= 0.65
        elif atm_class == "giant":
            score *= 0.30

        cap = self.SCORE_CAP.get(atm_class, 100.0)
        return min(score, cap)

    # ---------------------------------------------------
    # ATMOSPHERIC PROFILE
    # ---------------------------------------------------

    def _build_profile(self, g: Dict[str, float],
        transmission_data: Optional[Dict[str, float]] = None) -> AtmosphericProfile:

        td        = transmission_data or {}
        tox       = self._toxicity_index(g)
        ghi       = self._greenhouse_heating_index(g)
        density   = self._atmospheric_density(g, td)
        stability = self._thermal_stability(g)
        temp      = self._temperature_potential(ghi, td)

        return AtmosphericProfile(
            planet_type                = self._planet_type(g, td),
            dominant_gas_fingerprint   = self._fingerprint(g),
            greenhouse_intensity_label = self._greenhouse_label(ghi),
            greenhouse_heating_index   = round(ghi, 3),
            atmospheric_density        = density,
            thermal_stability          = stability,
            temperature_potential      = temp,
            toxicity_index             = round(tox, 4),
            toxicity_label             = self._toxicity_label(tox),
            atmosphere_similarity      = self._atmosphere_similarity(g),
        )

    # ---------------------------------------------------
    # GREENHOUSE HEATING INDEX  (0–1)
    # ---------------------------------------------------

    def _greenhouse_heating_index(self, g: Dict[str, float]) -> float:
        co2 = max(g.get("CO2", 0), 1e-9)
        ratio = max(co2 / self._CO2_REF, 1e-9)
        co2_forcing = min(1.0, math.log(ratio) / math.log(1000)) * 0.45

        h2o_forcing = min(1.0, g.get("H2O", 0) / 0.03) * 0.25
        ch4_forcing = min(1.0, g.get("CH4", 0) / 0.002) * 0.15
        o3_forcing  = min(1.0, g.get("O3",  0) / 0.00002) * 0.05

        return min(1.0, co2_forcing + h2o_forcing + ch4_forcing + o3_forcing)

    # ---------------------------------------------------
    # TOXICITY INDEX  (0–1)
    # ---------------------------------------------------

    def _toxicity_index(self, g: Dict[str, float]) -> float:

        tox = 0.0
        for gas, params in self._TOXICITY_PARAMS.items():
            conc = g.get(gas, 0)
            tox += min(1.0, conc / params["idlh"]) * params["weight"]
        return min(1.0, tox)

    # ---------------------------------------------------
    # PROFILE 
    # ---------------------------------------------------

    def _atmospheric_density(self, g: Dict[str, float],
                             td: Dict[str, float] = {}) -> str:

        h2_he = g.get("H2", 0) + g.get("HE", 0)
        if h2_he > 0.80:
            return "Low"

        mw = (
            g.get("CO2", 0) * 44 + g.get("N2",  0) * 28 + g.get("O2",  0) * 32 +
            g.get("H2O", 0) * 18 + g.get("CH4", 0) * 16 + g.get("H2",  0) *  2 +
            g.get("HE",  0) *  4 + g.get("SO2", 0) * 64 + g.get("CO",  0) * 28 +
            g.get("NH3", 0) * 17 + g.get("H2S", 0) * 34 + g.get("O3",  0) * 48
        )

        if g.get("CO2", 0) > 0.10 or g.get("H2O", 0) > 0.10:
            return "High"
        if mw > 25:
            return "High"
        if mw > 12:
            return "Medium"
        return "Low"

    def _thermal_stability(self, g: Dict[str, float]) -> str:
        instability = g.get("CH4", 0) * g.get("O2", 0) * 1000
        if instability > 2.0:
            return "Unstable"
        if instability > 0.5:
            return "Moderate"
        return "Stable"

    def _temperature_potential(self, ghi: float,
                               td: Dict[str, float] = {}) -> str:

        if ghi > 0.80:
            return "Extreme Heat"
        if ghi > 0.45:
            return "Warm"
        if ghi > 0.15:
            return "Moderate"
        if ghi > 0.05:
            return "Cool"
        return "Cold"

    def _planet_type(self, g: Dict[str, float],
                     td: Dict[str, float] = {}) -> str:

        planet_rad = td.get("mean_planet_radius", 0)
        h2_he = g.get("H2", 0) + g.get("HE", 0)

        if planet_rad > 1.0 or h2_he > 0.85:
            return "Gas Giant"

        if planet_rad > 0.35 or h2_he > 0.50:
            if g.get("CO2", 0) > 0.50:
                return "Venus-like (CO2-dominated)"
            if g.get("H2O", 0) > 0.02 and h2_he > 0.30:
                return "Hycean World Candidate"
            return "Ice Giant / Sub-Neptune"

        if planet_rad > 0.15:
            if g.get("CO2", 0) > 0.70:
                return "Venus-like (CO2-dominated)"
            if g.get("N2", 0) > 0.50 and g.get("O2", 0) > 0.10:
                return "Earth-like"
            if g.get("SO2", 0) > 0.01 or g.get("H2S", 0) > 0.01:
                return "Volcanically Active Rocky"
            return "Rocky / Mixed Atmosphere"

        if g.get("CO2", 0) > 0.70:
            return "Venus-like (CO2-dominated)"
        if g.get("N2", 0) > 0.50 and g.get("O2", 0) > 0.10:
            return "Earth-like"
        if g.get("N2", 0) > 0.80:
            return "Titan-like (N2-dominated)"
        return "Rocky / Mixed Atmosphere"

    def _fingerprint(self, g: Dict[str, float]) -> str:
        top = sorted(g.items(), key=lambda x: x[1], reverse=True)[:3]
        return ", ".join(f"{k} {v*100:.1f}%" for k, v in top)

    def _greenhouse_label(self, ghi: float) -> str:
        if ghi < 0.05:  return "Low"
        if ghi < 0.20:  return "Moderate"
        if ghi < 0.50:  return "High"
        return "Extreme"

    def _toxicity_label(self, val: float) -> str:
        if val < 0.10:  return "Low"
        if val < 0.30:  return "Moderate"
        if val < 0.60:  return "High"
        return "Extreme"

    # ---------------------------------------------------
    # ATMOSPHERE SIMILARITY  (log-weighted cosine)
    # ---------------------------------------------------

    def _similarity(self, a: Dict[str, float], b: Dict[str, float]) -> float:
        def vector(d):
            return [math.log1p(d.get(k, 0) * 10000) for k in self.GAS_KEYS]

        va = vector(a)
        vb = vector(b)
        dot = sum(x * y for x, y in zip(va, vb))
        mag_a = math.sqrt(sum(x * x for x in va))
        mag_b = math.sqrt(sum(y * y for y in vb))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    def _atmosphere_similarity(self, g: Dict[str, float]) -> List[AtmosphereSimilarity]:
        sims = [
            AtmosphereSimilarity(
                planet     = p,
                similarity = round(self._similarity(g, ref) * 100, 1),
            )
            for p, ref in self.SOLAR_SYSTEM.items()
        ]
        sims.sort(key=lambda x: x.similarity, reverse=True)
        return sims

    # ---------------------------------------------------
    # GRADE / CATEGORY / SUMMARY
    # ---------------------------------------------------

    def _grade(self, score: float) -> str:
        if score >= 80:  return "A"
        if score >= 60:  return "B"
        if score >= 40:  return "C"
        if score >= 20:  return "D"
        return "E"

    def _category(self, score: float) -> str:
        if score >= 80:  return "Highly Habitable"
        if score >= 60:  return "Potentially Habitable"
        if score >= 40:  return "Marginally Habitable"
        if score >= 20:  return "Unlikely Habitable"
        return "Extremely Hostile"

    def _summary(self, score: float,
                 biosigs: List[BiosignatureResult],
                 factors: Dict[str, float]) -> str:

        detected = [b.name for b in biosigs if b.detected]
        limiting = [k for k, v in factors.items() if v < 0.4]

        return (
            f"Habitability score: {score:.1f}/100. "
            f"Biosignatures detected: {', '.join(detected) if detected else 'None'}. "
            f"Limiting factors: {', '.join(limiting) if limiting else 'None'}."
        )


def analyze_planet(gas_predictions: Dict[str, float],
                   transmission_data: Optional[Dict[str, float]] = None) -> HabitabilityResult:

    return BiosignatureDetector().detect(gas_predictions, transmission_data)