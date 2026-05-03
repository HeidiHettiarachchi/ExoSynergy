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
    greenhouse_effect: float  
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
        "CO":  {"idlh": 0.001200, "lc50": 0.005000, "weight": 0.35, "description": "Carbon monoxide - binds hemoglobin"},
        "SO2": {"idlh": 0.000100, "lc50": 0.002500, "weight": 0.25, "description": "Sulfur dioxide - respiratory irritant"},
        "H2S": {"idlh": 0.000300, "lc50": 0.004400, "weight": 0.20, "description": "Hydrogen sulfide - neurotoxin"},
        "NH3": {"idlh": 0.003000, "lc50": 0.009230, "weight": 0.15, "description": "Ammonia - corrosive and toxic"},
        "CO2": {"idlh": 0.100000, "lc50": 0.500000, "weight": 0.03, "description": "Carbon dioxide - asphyxiant at high levels"},
        "CH4": {"idlh": 0.050000, "lc50": 0.500000, "weight": 0.02, "description": "Methane - asphyxiant, explosive"},
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
        ghe       = self._greenhouse_effect(g, td)  # New greenhouse effect calculation

        return AtmosphericProfile(
            planet_type                = self._planet_type(g, td),
            dominant_gas_fingerprint   = self._fingerprint(g),
            greenhouse_intensity_label = self._greenhouse_label(ghi),
            greenhouse_heating_index   = round(ghi, 3),
            greenhouse_effect          = round(ghe, 3),  # New field
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

        try:
            # CO2 forcing: ~3.7 W/m² per doubling, reference 280 ppm
            co2_ppm = max(float(g.get("CO2", 0)) * 1e6, 1e-9)
            co2_forcing = max(0.0, 3.7 * math.log2(co2_ppm / 280.0) / 5.35)  

            # H2O forcing: Strong absorber, ~2.0 W/m² per 1% increase
            h2o_pct = float(g.get("H2O", 0)) * 100
            h2o_forcing = min(1.0, h2o_pct / 3.0) * 0.25  # Cap at 3% for normalization

            # CH4 forcing: ~0.5 W/m² per doubling, reference 0.7 ppm
            ch4_ppm = max(float(g.get("CH4", 0)) * 1e6, 1e-9)
            ch4_forcing = max(0.0, 0.5 * math.log2(ch4_ppm / 0.7) / 3.0) 

            # O3 forcing: ~0.4 W/m² per doubling, reference 30 ppb
            o3_ppb = max(float(g.get("O3", 0)) * 1e9, 1e-9)
            o3_forcing = max(0.0, 0.4 * math.log2(o3_ppb / 30.0) / 2.0)  

            # N2O forcing: ~0.2 W/m² per doubling, reference 270 ppb
            n2o_ppb = max(float(g.get("N2O", 0)) * 1e9, 1e-9)
            n2o_forcing = max(0.0, 0.2 * math.log2(n2o_ppb / 270.0) / 1.5) 

            total_forcing = co2_forcing + h2o_forcing + ch4_forcing + o3_forcing + n2o_forcing

            # Normalize to 0-1 scale (Earth's GHI ~0.3-0.4)
            return min(1.0, max(0.0, total_forcing / 2.0))
        except Exception as e:
            print(f"Error in GHI calculation: {e}, g={g}")
            return 0.0

    # ---------------------------------------------------
    # GREENHOUSE EFFECT STRENGTH (0–1) 
    # ---------------------------------------------------

    def _greenhouse_effect(self, g: Dict[str, float],
                          td: Dict[str, float] = {}) -> float:

        # Base greenhouse effect from CO2 and H2O
        co2_effect = min(1.0, g.get("CO2", 0) / 0.01) * 0.4  # CO2 dominates
        h2o_effect = min(1.0, g.get("H2O", 0) / 0.03) * 0.3  # H2O secondary

        # Additional greenhouse gases
        ch4_effect = min(1.0, g.get("CH4", 0) / 0.002) * 0.15
        n2o_effect = min(1.0, g.get("N2O", 0) / 1e-6) * 0.1
        o3_effect = min(1.0, g.get("O3", 0) / 2e-5) * 0.05

        # Atmospheric pressure effect (higher pressure = stronger greenhouse)
        pressure_factor = td.get("mean_pressure", 1.0)  
        pressure_effect = min(1.0, math.log10(pressure_factor) / 2.0) * 0.1

        total_effect = co2_effect + h2o_effect + ch4_effect + n2o_effect + o3_effect + pressure_effect

        return min(1.0, max(0.0, total_effect))

    # ---------------------------------------------------
    # TOXICITY INDEX  
    # ---------------------------------------------------

    def _toxicity_index(self, g: Dict[str, float]) -> float:
       
        toxicity_score = 0.0

        for gas, params in self._TOXICITY_PARAMS.items():
            conc = g.get(gas, 0)

            # Use IDLH for immediate danger assessment
            idlh_risk = min(1.0, conc / params["idlh"]) * params["weight"]

            # Use LC50 for lethal concentration assessment
            lc50_risk = min(1.0, conc / params["lc50"]) * params["weight"] * 0.5

            # Combine risks with emphasis on IDLH for acute toxicity
            gas_toxicity = (idlh_risk * 0.7) + (lc50_risk * 0.3)
            toxicity_score += gas_toxicity

        # Consider synergistic effects
        # CO + H2S = enhanced neurotoxicity
        co_h2s_synergy = g.get("CO", 0) * g.get("H2S", 0) * 1000 * 0.1

        # SO2 + NH3 = acid-base reactions forming aerosols
        so2_nh3_synergy = g.get("SO2", 0) * g.get("NH3", 0) * 5000 * 0.05

        total_toxicity = min(1.0, toxicity_score + co_h2s_synergy + so2_nh3_synergy)

        return total_toxicity

    # ---------------------------------------------------
    # PROFILE 
    # ---------------------------------------------------

    def _atmospheric_density(self, g: Dict[str, float],
                             td: Dict[str, float] = {}) -> str:
      
        # Calculate mean molecular weight
        mw = (
            g.get("H2",  0) *  2.0 + g.get("HE",  0) *  4.0 +
            g.get("N2",  0) * 28.0 + g.get("O2",  0) * 32.0 +
            g.get("CO2", 0) * 44.0 + g.get("H2O", 0) * 18.0 +
            g.get("CH4", 0) * 16.0 + g.get("CO",  0) * 28.0 +
            g.get("NH3", 0) * 17.0 + g.get("SO2", 0) * 64.0 +
            g.get("H2S", 0) * 34.0 + g.get("O3",  0) * 48.0 +
            g.get("N2O", 0) * 44.0
        )

        # Get environmental factors
        pressure = td.get("mean_pressure", 1.0)  # bar
        temp = td.get("mean_temperature", 288.0)  # K (Earth average)

        # Ideal gas law: density = (P * MW) / (R * T)
        # R = 8.314 J/mol·K, convert to kg/m³
        density = (pressure * 1e5 * mw) / (8.314 * temp) * 1e-3  # kg/m³

        # Categorize density
        if density < 0.1:
            return "Very Low (< 0.1 kg/m³)"  # Like Mars
        elif density < 0.5:
            return "Low (0.1-0.5 kg/m³)"  # Thin atmospheres
        elif density < 1.5:
            return "Moderate (0.5-1.5 kg/m³)"  # Earth-like
        elif density < 5.0:
            return "Dense (1.5-5.0 kg/m³)"  # Thick atmospheres
        else:
            return "Very Dense (> 5.0 kg/m³)"  # Venus-like or super-Earths

    def _thermal_stability(self, g: Dict[str, float]) -> str:
        
        # Primary instability: CH4 + O2 → CO2 + H2O
        ch4_o2_instability = g.get("CH4", 0) * g.get("O2", 0) * 1000

        # Secondary instabilities
        co_o2_instability = g.get("CO", 0) * g.get("O2", 0) * 500   # CO oxidation
        h2_o2_instability = g.get("H2", 0) * g.get("O2", 0) * 200   # H2 combustion
        nh3_o2_instability = g.get("NH3", 0) * g.get("O2", 0) * 100  # NH3 oxidation

        # Photochemical instabilities (UV-driven)
        o3_instability = g.get("O3", 0) * 1e6  # Ozone photolysis
        so2_instability = g.get("SO2", 0) * 1e4  # Sulfur chemistry

        total_instability = (
            ch4_o2_instability + co_o2_instability + h2_o2_instability +
            nh3_o2_instability + o3_instability + so2_instability
        )

        # Consider stabilizing factors
        n2_buffer = g.get("N2", 0) * 0.5  # N2 dilutes reactive species
        noble_gases = g.get("HE", 0) + g.get("AR", 0) * 0.3  # Inert diluents

        net_instability = total_instability / (1 + n2_buffer + noble_gases)

        if net_instability > 10.0:
            return "Highly Unstable (Rapid chemical reactions expected)"
        elif net_instability > 3.0:
            return "Moderately Unstable (Active photochemistry)"
        elif net_instability > 1.0:
            return "Marginally Stable (Some disequilibrium)"
        elif net_instability > 0.1:
            return "Stable (Minor disequilibrium)"
        else:
            return "Highly Stable (Near equilibrium)"

    def _temperature_potential(self, ghi: float,
                               td: Dict[str, float] = {}) -> str:
        
        # Stellar flux (relative to Earth)
        stellar_flux = td.get("stellar_flux", 1.0)  # Earth = 1.0

        # Estimate based on atmospheric composition
        base_albedo = 0.3  # Earth-like default
        if td.get("albedo", 0) > 0:
            base_albedo = td.get("albedo", 0.3)
        else:
            # Estimate from composition
            cloud_albedo = min(0.3, td.get("H2O", 0) * 3)  # Water clouds
            ice_albedo = min(0.2, td.get("CO2", 0) * 2) if ghi < 0.1 else 0
            base_albedo = 0.1 + cloud_albedo + ice_albedo

        # Greenhouse effect amplification
        greenhouse_factor = 1 + ghi * 3  # GHI scales greenhouse effect

        # Energy balance: T_eff^4 = (S/4) * (1-A) * (1 + G) / σ
        # Simplified: T ∝ sqrt( (S * (1-A) * (1+G)) )
        effective_temp_factor = math.sqrt(stellar_flux * (1 - base_albedo) * greenhouse_factor)

        # Convert to temperature range
        earth_equiv_temp = 255  # Effective temperature for Earth (K)
        estimated_temp = earth_equiv_temp * effective_temp_factor

        if estimated_temp > 1000:
            return "Extreme Heat (>1000K) - Likely molten or highly ionized atmosphere"
        elif estimated_temp > 500:
            return "Very Hot (500-1000K) - Intense stellar heating or greenhouse effects"
        elif estimated_temp > 373:
            return "Hot (373-500K) - Above water boiling point"
        elif estimated_temp > 323:
            return "Warm (323-373K) - Moderate atmospheric temperatures"
        elif estimated_temp > 273:
            return "Temperate (273-323K) - Liquid water possible (if pressure suitable)"
        elif estimated_temp > 200:
            return "Cool (200-273K) - Ice formation likely"
        elif estimated_temp > 100:
            return "Cold (100-200K) - Cryogenic conditions"
        else:
            return "Extreme Cold (<100K) - Frozen atmospheric state"

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
        return "Extreme - High greenhouse gas presence"

    def _toxicity_label(self, val: float) -> str:
        if val < 0.10:  return "Low - Non-breathable"
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