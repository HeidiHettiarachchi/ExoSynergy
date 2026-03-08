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

    _CO2_REF = 0.000280

    _TOXICITY_PARAMS = {
        "CO":  {"idlh": 0.001200, "weight": 0.40},   # 1200 ppm IDLH
        "SO2": {"idlh": 0.000100, "weight": 0.80},   # 100 ppm IDLH
        "H2S": {"idlh": 0.000300, "weight": 1.20},   # 300 ppm IDLH
        "NH3": {"idlh": 0.003000, "weight": 0.20},   # 300 ppm IDLH (also biogenic)
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
        g     = {k: max(0.0, v) for k, v in g.items()}
        total = sum(g.values())
        return g if total == 0 else {k: v / total for k, v in g.items()}

    # ---------------------------------------------------
    # BIOSIGNATURE DETECTION
    # ---------------------------------------------------

    def _detect_biosignatures(self, g: Dict[str, float], atm_class: str) -> List[BiosignatureResult]:

        if atm_class == "giant":
            return []

        biosigs = []

        if atm_class == "sub_neptune":
            h2o = g.get("H2O", 0)
            if h2o > 0.001:
                biosigs.append(BiosignatureResult(
                    name           = "Water Vapor",
                    detected       = True,
                    reason         = (
                        f"H2O at {h2o*100:.2f}% — possible Hycean world candidate; "
                        "water vapor detected in H2-rich envelope"
                    ),
                    gases_involved = ["H2O"],
                ))
            return biosigs

        o2  = g.get("O2",  0)
        ch4 = g.get("CH4", 0)
        if o2 > 0.01 and ch4 > 0.000010:   
            biosigs.append(BiosignatureResult(
                name           = "Oxygen-Methane Disequilibrium",
                detected       = True,
                reason         = (
                    f"O2 ({o2*100:.2f}%) and CH4 ({ch4*1e6:.1f} ppm) coexist — "
                    "thermodynamically unstable without active biological replenishment"
                ),
                gases_involved = ["O2", "CH4"],
            ))

        o3 = g.get("O3", 0)
        if o3 > 0.000001 and o2 > 0.001:  
            biosigs.append(BiosignatureResult(
                name           = "Ozone Shield",
                detected       = True,
                reason         = (
                    f"O3 at {o3*1e6:.2f} ppm — implies sustained O2 photochemistry "
                    "and UV shielding compatible with surface life"
                ),
                gases_involved = ["O3"],
            ))

        n2o = g.get("N2O", 0)
        if n2o > 0.0000001:   # > 0.1 ppb
            biosigs.append(BiosignatureResult(
                name           = "Nitrous Oxide",
                detected       = True,
                reason         = (
                    f"N2O at {n2o*1e9:.1f} ppb — almost exclusively produced by "
                    "microbial denitrification; strong biogenic indicator"
                ),
                gases_involved = ["N2O"],
            ))

        nh3 = g.get("NH3", 0)
        if nh3 > 0.000001 and g.get("H2", 0) < 0.10:  
            biosigs.append(BiosignatureResult(
                name           = "Ammonia Signature",
                detected       = True,
                reason         = (
                    f"NH3 at {nh3*1e6:.2f} ppm — photolytically unstable; "
                    "requires active biological or hydrothermal source"
                ),
                gases_involved = ["NH3"],
            ))

        h2o = g.get("H2O", 0)
        if h2o > 0.001:   # > 0.1%
            biosigs.append(BiosignatureResult(
                name           = "Water Vapor",
                detected       = True,
                reason         = (
                    f"H2O at {h2o*100:.2f}% — liquid water potential present; "
                    "universal solvent and prerequisite for known life"
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
        elif o2 <= 0.30:
            factors["oxygen"] = max(0.0, 1.0 - abs(o2 - 0.21) / 0.09)
        else:
            factors["oxygen"] = max(0.0, 1.0 - (o2 - 0.30) / 0.70)

        h2o = g.get("H2O", 0)
        if h2o <= 0.0:
            factors["water"] = 0.0
        elif h2o <= 0.04:
            factors["water"] = min(1.0, h2o / 0.01)
        else:
            factors["water"] = max(0.0, 1.0 - (h2o - 0.04) / 0.06)

        ghi = self._greenhouse_heating_index(g)
        if ghi < 0.005:
            factors["greenhouse_penalty"] = 0.30              
        elif ghi <= 0.20:
            factors["greenhouse_penalty"] = 0.30 + (ghi - 0.005) / 0.195 * 0.70
        elif ghi <= 0.40:
            factors["greenhouse_penalty"] = 1.0              
        elif ghi <= 0.80:
            factors["greenhouse_penalty"] = max(0.0, 1.0 - (ghi - 0.40) / 0.40)
        else:
            factors["greenhouse_penalty"] = 0.0          

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
       
        co2 = max(g.get("CO2", 0), 1e-10)
        co2_forcing = max(0.0, math.log(co2 / self._CO2_REF)) / 8.5  # ln(~100%/280ppm)≈8.5

        h2o_forcing = min(1.0, g.get("H2O", 0) / 0.05) * 0.25
        ch4_forcing = min(1.0, g.get("CH4", 0) / 0.001) * 0.15
        o3_forcing  = min(1.0, g.get("O3",  0) / 0.00001) * 0.05

        return min(1.0, co2_forcing * 0.55 + h2o_forcing + ch4_forcing + o3_forcing)

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
    # PROFILE HELPERS
    # ---------------------------------------------------

    def _atmospheric_density(self, g: Dict[str, float],
                             td: Dict[str, float] = {}) -> str:

        rad_ratio    = td.get("mean_rad_ratio", 0)
        planet_rad   = td.get("mean_planet_radius", 0)    
        stellar_rad  = td.get("mean_stellar_radius", 0)   
        transit_dep  = td.get("mean_transit_depth", 0)    
        if rad_ratio > 0 and planet_rad > 0:

            rp_rs_sq = rad_ratio ** 2
            depth_excess = transit_dep - rp_rs_sq if transit_dep > 0 else 0

            if planet_rad > 0.8:                    
                return "Low"                        
            elif planet_rad > 0.4:                 
                if depth_excess > 0.002:
                    return "Low"                    
                return "Medium"
            else:                                  
                if depth_excess > 0.001:
                    return "Medium"                 
                return "High"                       

        mw = (
            g.get("CO2", 0) * 44 + g.get("N2",  0) * 28 + g.get("O2",  0) * 32 +
            g.get("H2O", 0) * 18 + g.get("CH4", 0) * 16 + g.get("H2",  0) *  2 +
            g.get("HE",  0) *  4 + g.get("SO2", 0) * 64 + g.get("CO",  0) * 28 +
            g.get("NH3", 0) * 17 + g.get("H2S", 0) * 34 + g.get("O3",  0) * 48
        )
        if mw > 30:  return "High"
        if mw > 15:  return "Medium"
        return "Low"

    def _thermal_stability(self, g: Dict[str, float]) -> str:
        """O2-CH4 reactive instability proportional to product of both fractions."""
        instability = g.get("CH4", 0) * g.get("O2", 0) * 1000
        if instability > 5.0:  return "Unstable"
        if instability > 1.0:  return "Moderate"
        return "Stable"

    def _temperature_potential(self, ghi: float,
                               td: Dict[str, float] = {}) -> str:

        planet_rad = td.get("mean_planet_radius", 0)   

        if planet_rad > 0:
            if planet_rad > 1.5:
                return "Extreme Heat"
            elif planet_rad > 0.8:
                return "Warm"
            elif planet_rad > 0.35:
                if ghi > 0.40:   return "Warm"
                if ghi > 0.15:   return "Moderate"
                return "Cool"
            else:
                if ghi > 0.70:   return "Extreme Heat"
                if ghi > 0.40:   return "Warm"
                if ghi > 0.15:   return "Moderate"
                if ghi > 0.05:   return "Cool"
                return "Cold"

        if ghi > 0.70:   return "Extreme Heat"
        if ghi > 0.40:   return "Warm"
        if ghi > 0.15:   return "Moderate"
        if ghi > 0.05:   return "Cool"
        return "Cold"

    def _planet_type(self, g: Dict[str, float],
                     td: Dict[str, float] = {}) -> str:

        planet_rad = td.get("mean_planet_radius", 0)  

        if planet_rad > 0:
            h2_he = g.get("H2", 0) + g.get("HE", 0)

            if planet_rad > 1.0:
                return "Gas Giant"

            elif planet_rad > 0.35:
                if h2_he > 0.50 and g.get("H2O", 0) > 0.02:
                    return "Hycean World Candidate"
                if g.get("CO2", 0) > 0.50:
                    return "Venus-like (CO2-dominated)"
                return "Ice Giant / Sub-Neptune"

            elif planet_rad > 0.15:
                if g.get("CO2", 0) > 0.70:
                    return "Venus-like (CO2-dominated)"
                if g.get("N2", 0) > 0.50 and g.get("O2", 0) > 0.10:
                    return "Earth-like"
                if g.get("SO2", 0) > 0.01 or g.get("H2S", 0) > 0.01:
                    return "Volcanically Active Rocky"
                return "Rocky / Mixed Atmosphere"

            else:
                return "Rocky / Mixed Atmosphere"

        h2_he = g.get("H2", 0) + g.get("HE", 0)
        if h2_he > 0.85:
            return "Gas Giant"
        if h2_he > 0.60:
            return "Hycean World Candidate" if g.get("H2O", 0) > 0.02 else "Ice Giant / Sub-Neptune"
        if g.get("CO2", 0) > 0.70:
            return "Venus-like (CO2-dominated)"
        if g.get("N2", 0) > 0.50 and g.get("O2", 0) > 0.10:
            return "Earth-like"
        if g.get("N2", 0) > 0.80:
            return "Titan-like (N2-dominated)"
        if g.get("SO2", 0) > 0.01 or g.get("H2S", 0) > 0.01:
            return "Volcanically Active Rocky"
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
       
        keys  = set(a) | set(b)
        def lw(d, k): return math.log1p(d.get(k, 0) * 1000)
        dot   = sum(lw(a, k) * lw(b, k) for k in keys)
        mag_a = math.sqrt(sum(lw(a, k) ** 2 for k in keys))
        mag_b = math.sqrt(sum(lw(b, k) ** 2 for k in keys))
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


# ---------------------------------------------------
# PUBLIC FUNCTION
# ---------------------------------------------------

def analyze_planet(gas_predictions: Dict[str, float],
                   transmission_data: Optional[Dict[str, float]] = None) -> HabitabilityResult:

    return BiosignatureDetector().detect(gas_predictions, transmission_data)