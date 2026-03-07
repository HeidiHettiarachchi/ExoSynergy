from dataclasses import dataclass
from typing import Dict, List, Optional
import math


# ---------------------------------------------------
# DATA STRUCTURES  — field names unchanged
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


# ---------------------------------------------------
# MAIN DETECTOR
# ---------------------------------------------------

class BiosignatureDetector:

    # Reference atmospheres — mole fractions (NASA Planetary Fact Sheets)
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

    # Factor keys match original exactly
    FACTOR_WEIGHTS = {
        "oxygen":             0.30,
        "water":              0.25,
        "greenhouse_penalty": 0.20,
        "low_toxicity":       0.15,
        "nitrogen_buffer":    0.10,
    }

    # Biosignature bonus scores — weighted by scientific consensus strength
    BIOSIG_WEIGHTS = {
        "Oxygen-Methane Disequilibrium": 15,
        "Ozone Shield":                  10,
        "Nitrous Oxide":                 10,
        "Ammonia Signature":              5,
        "Water Vapor":                    3,
    }

    # Planetary class score caps
    SCORE_CAP = {
        "giant":       10.0,
        "sub_neptune": 35.0,
        "rocky":       100.0,
    }

    # CO2 pre-industrial reference (280 ppm) for log forcing
    _CO2_REF = 0.000280

    # NIOSH IDLH-based toxicity thresholds (mole fractions)
    # https://www.cdc.gov/niosh/idlh/
    _TOXICITY_PARAMS = {
        "CO":  {"idlh": 0.001200, "weight": 0.40},   # 1200 ppm IDLH
        "SO2": {"idlh": 0.000100, "weight": 0.80},   # 100 ppm IDLH
        "H2S": {"idlh": 0.000300, "weight": 1.20},   # 300 ppm IDLH
        "NH3": {"idlh": 0.003000, "weight": 0.20},   # 300 ppm IDLH (also biogenic)
    }

    # ---------------------------------------------------
    # PUBLIC ENTRY
    # ---------------------------------------------------

    def detect(self, gases: Dict[str, float],
               transmission_data: Optional[Dict[str, float]] = None) -> HabitabilityResult:
        """
        Parameters
        ----------
        gases : dict
            Gas mole fractions from ML model prediction.
        transmission_data : dict, optional
            Physical parameters from transmission spectroscopy preprocessing.
            Expected keys (all optional):
              mean_planet_radius      – planet radius in Jupiter radii (PL_RADJ)
              mean_stellar_radius     – stellar radius in solar radii  (ST_RAD)
              mean_rad_ratio          – Rp/Rs radius ratio             (PL_RATROR)
              mean_transit_depth      – transit depth fraction         (PL_TRANDEP)
              mean_transit_depth_uncertainty
              mean_radius_uncertainty
            When provided, atmospheric_density, planet_type, and
            temperature_potential are refined using physical constraints.
            Eclipse and direct-imaging paths pass None → original logic unchanged.
        """
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

        # Gas giants cannot support surface life and have abundant abiotic CH4,
        # NH3, and H2O — biosignatures are physically meaningless here.
        if atm_class == "giant":
            return []

        biosigs = []

        # Sub-Neptunes: only Water Vapor is meaningful (Hycean world context).
        # O2/O3/N2O/NH3 checks are suppressed — H2-dominated atmospheres
        # produce abiotic NH3 and lack surface UV shielding context for O3.
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

        # ── Rocky worlds only below this point ────────────────────────────────

        # ── Oxygen-Methane Disequilibrium ──────────────────────────────────────
        # O2 + CH4 react rapidly (τ ≈ 10 yr). Coexistence requires continuous
        # biological replenishment. Thresholds: Schwieterman et al. 2018.
        o2  = g.get("O2",  0)
        ch4 = g.get("CH4", 0)
        if o2 > 0.01 and ch4 > 0.000010:   # O2 > 1%, CH4 > 10 ppm
            biosigs.append(BiosignatureResult(
                name           = "Oxygen-Methane Disequilibrium",
                detected       = True,
                reason         = (
                    f"O2 ({o2*100:.2f}%) and CH4 ({ch4*1e6:.1f} ppm) coexist — "
                    "thermodynamically unstable without active biological replenishment"
                ),
                gases_involved = ["O2", "CH4"],
            ))

        # ── Ozone Shield ───────────────────────────────────────────────────────
        # O3 > 1 ppm implies sustained oxygenic photosynthesis source.
        # Threshold: Segura et al. 2005.
        # Minimum O2 guard: O3 without meaningful O2 is abiotic photochemistry.
        o3 = g.get("O3", 0)
        if o3 > 0.000001 and o2 > 0.001:   # O3 > 1 ppm AND O2 > 0.1%
            biosigs.append(BiosignatureResult(
                name           = "Ozone Shield",
                detected       = True,
                reason         = (
                    f"O3 at {o3*1e6:.2f} ppm — implies sustained O2 photochemistry "
                    "and UV shielding compatible with surface life"
                ),
                gases_involved = ["O3"],
            ))

        # ── Nitrous Oxide ──────────────────────────────────────────────────────
        # N2O has no significant abiotic source on rocky worlds.
        # Threshold: Sagan et al. 1993, Rugheimer et al. 2015.
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

        # ── Ammonia Signature ──────────────────────────────────────────────────
        # NH3 is photolyzed rapidly on rocky worlds; persistence requires active source.
        # Guard: suppress if H2 > 10% — abiotic NH3 is expected in H2-rich envelopes.
        nh3 = g.get("NH3", 0)
        if nh3 > 0.000001 and g.get("H2", 0) < 0.10:   # > 1 ppm, low H2 context
            biosigs.append(BiosignatureResult(
                name           = "Ammonia Signature",
                detected       = True,
                reason         = (
                    f"NH3 at {nh3*1e6:.2f} ppm — photolytically unstable; "
                    "requires active biological or hydrothermal source"
                ),
                gases_involved = ["NH3"],
            ))

        # ── Water Vapor ────────────────────────────────────────────────────────
        # Necessary (not sufficient). Threshold > 1000 ppm detectable.
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
    # FACTOR SCORING  (each → 0.0–1.0)
    # ---------------------------------------------------

    def _score_factors(self, g: Dict[str, float], atm_class: str) -> Dict[str, float]:

        factors = {}

        # ── oxygen ─────────────────────────────────────────────────────────────
        # Bell-curve centred on 0.21 (Earth). <5% hypoxic; >30% fire/oxidative risk.
        o2 = g.get("O2", 0)
        if o2 <= 0.0:
            factors["oxygen"] = 0.0
        elif o2 < 0.05:
            factors["oxygen"] = (o2 / 0.05) * 0.3
        elif o2 <= 0.30:
            factors["oxygen"] = max(0.0, 1.0 - abs(o2 - 0.21) / 0.09)
        else:
            factors["oxygen"] = max(0.0, 1.0 - (o2 - 0.30) / 0.70)

        # ── water ──────────────────────────────────────────────────────────────
        # Optimal: 0.1–4% (Earth troposphere ≈ 1%). >5% → runaway risk.
        h2o = g.get("H2O", 0)
        if h2o <= 0.0:
            factors["water"] = 0.0
        elif h2o <= 0.04:
            factors["water"] = min(1.0, h2o / 0.01)
        else:
            factors["water"] = max(0.0, 1.0 - (h2o - 0.04) / 0.06)

        # ── greenhouse_penalty ─────────────────────────────────────────────────
        # Inverted from GHI: moderate warming → 1.0; extreme or none → 0.
        # Uses logarithmic CO2 forcing (IPCC AR6 style).
        ghi = self._greenhouse_heating_index(g)
        if ghi < 0.005:
            factors["greenhouse_penalty"] = 0.30              # Too cold
        elif ghi <= 0.20:
            factors["greenhouse_penalty"] = 0.30 + (ghi - 0.005) / 0.195 * 0.70
        elif ghi <= 0.40:
            factors["greenhouse_penalty"] = 1.0               # Sweet spot
        elif ghi <= 0.80:
            factors["greenhouse_penalty"] = max(0.0, 1.0 - (ghi - 0.40) / 0.40)
        else:
            factors["greenhouse_penalty"] = 0.0               # Runaway

        # ── low_toxicity ───────────────────────────────────────────────────────
        tox = self._toxicity_index(g)
        factors["low_toxicity"] = max(0.0, 1.0 - tox)

        # ── nitrogen_buffer ────────────────────────────────────────────────────
        # Optimal 0.70–0.80 (Earth = 0.78). Penalises excess above 0.78.
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
        """
        Physically grounded GHI:
        - CO2: logarithmic forcing ΔF = 5.35 × ln(C/C₀), IPCC AR6,
          normalised over [280 ppm → ~100% CO2] → 0–1
        - H2O, CH4, O3: linear contributions capped at realistic maxima
        """
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
        """
        NIOSH IDLH-grounded score. Each gas contributes proportionally
        to how far its concentration exceeds its safe threshold.
        """
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
        """
        Transmission path: use transit depth and radius ratio to estimate
        atmospheric scale height proxy → more physically accurate density class.

        Transit depth = (Rp/Rs)² → tells us how much atmosphere blocks light.
        Radius ratio (Rp/Rs) combined with planet radius gives bulk density proxy.

        Eclipse / direct-imaging path (td empty): falls back to mean molecular
        weight calculation as before.
        """
        rad_ratio    = td.get("mean_rad_ratio", 0)
        planet_rad   = td.get("mean_planet_radius", 0)    # Jupiter radii
        stellar_rad  = td.get("mean_stellar_radius", 0)   # Solar radii
        transit_dep  = td.get("mean_transit_depth", 0)    # Fraction

        if rad_ratio > 0 and planet_rad > 0:
            # Bulk density proxy: larger planet with small radius ratio → puffy / low density
            # Jupiter = 1.0 Rj. Rocky super-Earths typically < 0.3 Rj.
            # Scale height signal: high transit depth relative to radius ratio²
            # suggests an extended, low-density atmosphere.
            rp_rs_sq = rad_ratio ** 2
            depth_excess = transit_dep - rp_rs_sq if transit_dep > 0 else 0

            if planet_rad > 0.8:                    # Jupiter-sized or larger
                return "Low"                        # H2/He dominated, low mean MW
            elif planet_rad > 0.4:                  # Saturn–Neptune range
                if depth_excess > 0.002:
                    return "Low"                    # Extended puffy atmosphere
                return "Medium"
            else:                                   # Super-Earth / rocky range
                if depth_excess > 0.001:
                    return "Medium"                 # Some atmospheric puffiness
                return "High"                       # Dense rocky atmosphere

        # ── Fallback: mean molecular weight (eclipse / direct imaging) ─────────
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
        """
        Transmission path: planet radius gives a size-based temperature class.
        Larger planets (Jovian) are gas giants — intrinsically hot from formation
        or irradiation. Smaller rocky planets rely on greenhouse effect alone.

        Combined approach: radius sets a floor/ceiling, GHI refines within rocky range.
        Eclipse / direct-imaging (td empty): GHI-only as before.
        """
        planet_rad = td.get("mean_planet_radius", 0)   # Jupiter radii

        if planet_rad > 0:
            if planet_rad > 1.5:
                # Hot Jupiter / super-Jupiter — always hot regardless of GHI
                # (irradiation + internal heat dominate)
                return "Extreme Heat"
            elif planet_rad > 0.8:
                # Jupiter / Saturn scale — warm from irradiation + gravity
                return "Warm"
            elif planet_rad > 0.35:
                # Neptune / sub-Neptune — moderate, GHI still matters
                if ghi > 0.40:   return "Warm"
                if ghi > 0.15:   return "Moderate"
                return "Cool"
            else:
                # Rocky / super-Earth — GHI is the primary driver
                if ghi > 0.70:   return "Extreme Heat"
                if ghi > 0.40:   return "Warm"
                if ghi > 0.15:   return "Moderate"
                if ghi > 0.05:   return "Cool"
                return "Cold"

        # ── Fallback: GHI-only (eclipse / direct imaging) ─────────────────────
        if ghi > 0.70:   return "Extreme Heat"
        if ghi > 0.40:   return "Warm"
        if ghi > 0.15:   return "Moderate"
        if ghi > 0.05:   return "Cool"
        return "Cold"

    def _planet_type(self, g: Dict[str, float],
                     td: Dict[str, float] = {}) -> str:
        """
        Transmission path: planet radius (PL_RADJ) provides direct physical
        classification that overrides the gas-composition heuristic.
        Radius boundaries from Fulton gap / Chen & Kipping 2017:
          > 1.0  Rj  → Gas Giant
          0.35–1.0 Rj → Sub-Neptune / Ice Giant
          0.15–0.35 Rj → Super-Earth
          < 0.15 Rj   → Rocky Earth-scale
        Eclipse / direct-imaging (td empty): composition heuristic as before.
        """
        planet_rad = td.get("mean_planet_radius", 0)   # Jupiter radii

        if planet_rad > 0:
            h2_he = g.get("H2", 0) + g.get("HE", 0)

            if planet_rad > 1.0:
                return "Gas Giant"

            elif planet_rad > 0.35:
                # Sub-Neptune range — check for Hycean signature
                if h2_he > 0.50 and g.get("H2O", 0) > 0.02:
                    return "Hycean World Candidate"
                if g.get("CO2", 0) > 0.50:
                    return "Venus-like (CO2-dominated)"
                return "Ice Giant / Sub-Neptune"

            elif planet_rad > 0.15:
                # Super-Earth range
                if g.get("CO2", 0) > 0.70:
                    return "Venus-like (CO2-dominated)"
                if g.get("N2", 0) > 0.50 and g.get("O2", 0) > 0.10:
                    return "Earth-like"
                if g.get("SO2", 0) > 0.01 or g.get("H2S", 0) > 0.01:
                    return "Volcanically Active Rocky"
                return "Rocky / Mixed Atmosphere"

            else:
                # Small rocky world
                return "Rocky / Mixed Atmosphere"

        # ── Fallback: composition heuristic (eclipse / direct imaging) ─────────
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
        """
        Log-weighted cosine similarity.
        log1p prevents dominant gases from masking trace biosignature differences
        — critical for distinguishing Earth-like from Mars-like atmospheres.
        """
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
    """
    Parameters
    ----------
    gas_predictions : dict
        Gas mole fractions from ML model.
    transmission_data : dict, optional
        Pass the output of preprocess_transmission() here for transmission
        spectroscopy observations. Leave as None for eclipse or direct imaging
        — those paths use composition-only logic unchanged.

    Example
    -------
    # Transmission observation:
    result = analyze_planet(gas_preds, transmission_data=preprocess_transmission(df))

    # Eclipse or direct imaging observation:
    result = analyze_planet(gas_preds)
    """
    return BiosignatureDetector().detect(gas_predictions, transmission_data)