def apply_physics_adjustments(major, spectral_features, astro_features, data_type):

    adjusted = major.copy()

    radius = astro_features.get("PL_RADJ", 1.0)      # Jupiter radii
    temp   = astro_features.get("PL_EQTK", 1000.0)   # Equilibrium temperature (K)
    mass   = astro_features.get("PL_MASSJ", 1.0)      # Jupiter masses

    # -------------------------------------------------------
    # 
    # Gas giant (radius > 0.5 Rj, mass > 0.1 Mj)
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
    # Sub-Neptune / mini-Neptune (0.1 < radius < 0.5)
    # -------------------------------------------------------
    elif 0.1 < radius < 0.5:
        # Can have significant H2/He but also heavier molecules
        adjusted["H2O"] = max(adjusted.get("H2O", 0), 0.1)
        adjusted["CH4"] = max(adjusted.get("CH4", 0), 0.01)

    # -------------------------------------------------------
    # Observational type adjustments
    # -------------------------------------------------------
    if data_type == "transmission":
        # Transmission spectra are excellent for molecular absorption features
        # Boost gases commonly detected in transmission
        adjusted["H2O"] = max(adjusted.get("H2O", 0), 0.01)  # Water vapor
        adjusted["CH4"] = max(adjusted.get("CH4", 0), 0.005)  # Methane
        adjusted["CO2"] = max(adjusted.get("CO2", 0), 0.001)  # CO2
        adjusted["NH3"] = max(adjusted.get("NH3", 0), 0.001)  # Ammonia

    elif data_type == "eclipse":
        # Secondary eclipse spectra show thermal emission
        # Boost gases that emit in thermal IR
        adjusted["CO"] = max(adjusted.get("CO", 0), 0.01)   # Carbon monoxide
        adjusted["H2O"] = max(adjusted.get("H2O", 0), 0.01)  # Water
        adjusted["CH4"] = max(adjusted.get("CH4", 0), 0.005) # Methane
        # Suppress gases less prominent in emission
        adjusted["N2"] = min(adjusted.get("N2", 0), 0.1)  # N2 weak emitter

    elif data_type == "direct":
        # Direct imaging often for young, hot planets
        # Can detect a wide range, but often H2O, CO, CH4
        adjusted["H2O"] = max(adjusted.get("H2O", 0), 0.01)
        adjusted["CO"] = max(adjusted.get("CO", 0), 0.01)
        adjusted["CH4"] = max(adjusted.get("CH4", 0), 0.005)

    # -------------------------------------------------------
    # Enforce minimum physically acceptable compositions
    # -------------------------------------------------------
    if radius >= 0.5 and mass >= 0.1:
        # Gas giants: ensure H2 + He >= 60% if predicted is low
        h2_he = adjusted.get("H2", 0) + adjusted.get("He", 0)
        if h2_he < 60:
            adjusted["H2"] = max(adjusted.get("H2", 0), 50.0)
            adjusted["He"] = max(adjusted.get("He", 0), 10.0)
            # Scale down other gases
            others = {k: v for k, v in adjusted.items() if k not in ["H2", "He"]}
            others_total = sum(others.values())
            if others_total > 10:
                scale = 10 / others_total
                for k in others:
                    adjusted[k] *= scale

    elif 0.1 < radius < 0.5:
        # Sub-Neptunes: ensure H2 + He >= 40% if low
        h2_he = adjusted.get("H2", 0) + adjusted.get("He", 0)
        if h2_he < 40:
            adjusted["H2"] = max(adjusted.get("H2", 0), 30.0)
            adjusted["He"] = max(adjusted.get("He", 0), 10.0)
        adjusted["H2O"] = max(adjusted.get("H2O", 0), 5.0)

    else:
        # Rocky planets: ensure N2 + O2 >= 90% if low
        n2_o2 = adjusted.get("N2", 0) + adjusted.get("O2", 0)
        if n2_o2 < 90:
            adjusted["N2"] = max(adjusted.get("N2", 0), 78.0)
            adjusted["O2"] = max(adjusted.get("O2", 0), 21.0)
            # Scale down other gases
            others = {k: v for k, v in adjusted.items() if k not in ["N2", "O2"]}
            others_total = sum(others.values())
            if others_total > 1:
                scale = 1 / others_total
                for k in others:
                    adjusted[k] *= scale

    # -------------------------------------------------------
    # ALWAYS renormalize so percentages sum to 100%
    # -------------------------------------------------------
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {k: v / total * 100.0 for k, v in adjusted.items()}

    return adjusted