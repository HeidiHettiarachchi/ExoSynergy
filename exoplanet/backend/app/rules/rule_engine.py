def apply_physics_adjustments(major, spectral_features, astro_features, data_type):
    """
    Apply physically motivated post-prediction adjustments.
    Uses radius AND temperature to determine regime.
    Always renormalizes to 100% after any adjustment.
    """
    adjusted = major.copy()

    radius = astro_features.get("PL_RADJ", 1.0)      # Jupiter radii
    temp   = astro_features.get("PL_EQTK", 1000.0)   # Equilibrium temperature (K)
    mass   = astro_features.get("PL_MASSJ", 1.0)      # Jupiter masses

    # -------------------------------------------------------
    # REGIME 1: Gas giant (radius > 0.5 Rj, mass > 0.1 Mj)
    # -------------------------------------------------------
    if radius >= 0.5 and mass >= 0.1:
        # Suppress terrestrial gases that shouldn't dominate
        adjusted["N2"]  = min(adjusted.get("N2", 0),  0.5)   # N2 shouldn't be >0.5% in gas giant
        adjusted["O2"]  = min(adjusted.get("O2", 0),  0.01)  # O2 trace only
        adjusted["O3"]  = min(adjusted.get("O3", 0),  0.001)

        # Boost H2/He to compensate
        adjusted["H2"] *= 1.05
        adjusted["He"] *= 1.02

        # Cool gas giants (T < 800K): expect CH4, NH3, H2S
        if temp < 800:
            adjusted["CH4"] = max(adjusted.get("CH4", 0), 0.05)
            adjusted["NH3"] = max(adjusted.get("NH3", 0), 0.01)
            adjusted["H2S"] = max(adjusted.get("H2S", 0), 0.005)

        # Hot gas giants (T > 1200K): CO dominates over CH4
        if temp > 1200:
            adjusted["CO"]  = max(adjusted.get("CO",  0), 0.05)
            adjusted["CH4"] = min(adjusted.get("CH4", 0), 0.001)  # CH4 thermally dissociates

        # Water: should always be detectable in warm gas giants
        if temp > 500:
            adjusted["H2O"] = max(adjusted.get("H2O", 0), 0.05)

    # -------------------------------------------------------
    # REGIME 2: Sub-Neptune / mini-Neptune (0.1 < radius < 0.5)
    # -------------------------------------------------------
    elif 0.1 < radius < 0.5:
        # Can have significant H2/He but also heavier molecules
        adjusted["H2O"] = max(adjusted.get("H2O", 0), 0.1)
        adjusted["CH4"] = max(adjusted.get("CH4", 0), 0.01)

    # -------------------------------------------------------
    # REGIME 3: Rocky / terrestrial (radius < 0.1 Rj ~ <1.1 Re)
    # -------------------------------------------------------
    else:
        adjusted["N2"] *= 1.1
        adjusted["O2"] *= 1.05
        # Suppress H2/He — rocky planets lose hydrogen
        adjusted["H2"] = min(adjusted.get("H2", 0), 5.0)
        adjusted["He"] = min(adjusted.get("He", 0), 2.0)

    # -------------------------------------------------------
    # ALWAYS renormalize so percentages sum to 100%
    # -------------------------------------------------------
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {k: v / total * 100.0 for k, v in adjusted.items()}

    return adjusted