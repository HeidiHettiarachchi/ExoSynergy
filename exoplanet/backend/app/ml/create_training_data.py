import h5py
import pandas as pd
import numpy as np
from collections import Counter


def create_training_dataset_with_groundtruth(
    hdf5_spectra_path,
    csv_truth_path,
    output_h5_path,
    sample_fraction=1.0
):

    print("\nCreating the training dataset\n")

    truth_df = pd.read_csv(csv_truth_path)

    with h5py.File(hdf5_spectra_path, 'r') as f:

        spectrum_groups = sorted(list(f.keys()))

        # Extract planet IDs
        planet_ids = []
        for gname in spectrum_groups:
            if gname.startswith('Planet_'):
                try:
                    pid = int(gname.split('_')[1])
                    planet_ids.append(pid)
                except:
                    pass

        id_to_group = {pid: f'Planet_{pid}' for pid in planet_ids}

        truth_subset = truth_df[truth_df['planet_ID'].isin(planet_ids)].copy()
        truth_subset = truth_subset.sort_values('planet_ID').reset_index(drop=True)

        # Sampling
        if sample_fraction < 1.0:
            truth_subset = truth_subset.sample(frac=sample_fraction, random_state=42)

        specs = []
        labels = []
        regimes = []

        # 🔥 MAIN LOOP (file already open)
        for _, row in truth_subset.iterrows():

            planet_id = int(row['planet_ID'])
            group_name = id_to_group[planet_id]

            try:
                g = f[group_name]

                spectrum = g['instrument_spectrum'][()].flatten()
                width = g['instrument_width'][()].flatten()
                noise = g['instrument_noise'][()].flatten()
                wl = g['instrument_wlgrid'][()].flatten()

                # 🔹 Spectral features
                grad = np.gradient(spectrum)
                curvature = np.gradient(grad)

                features = np.concatenate([
                    spectrum,
                    grad,
                    curvature,
                    width,
                    noise,
                    wl
                ]).astype(np.float32)

                # 🔹 Normalize features (important for ML)
                features = (features - np.mean(features)) / (np.std(features) + 1e-8)

                specs.append(features)

                # -----------------------------
                # REAL CHEMISTRY (from CSV)
                # -----------------------------
                log_cols = ['log_H2O', 'log_CO2', 'log_CH4', 'log_CO', 'log_NH3']

                chem_vmr = np.array([
                    10.0 ** row[col] if row[col] > -30 else 1e-30
                    for col in log_cols
                ])

                h2o, co2, ch4, co, nh3 = chem_vmr

                # -----------------------------
                # Regime classification
                # -----------------------------
                reducing = ch4 + nh3
                oxidized = co2
                water = h2o

                if reducing > oxidized and reducing > water:
                    regime = 0
                elif oxidized > reducing and oxidized > water:
                    regime = 1
                else:
                    regime = 2

                regimes.append(regime)

                total_score = reducing + oxidized + water + 1e-12

                r_w = reducing / total_score
                o_w = oxidized / total_score
                w_w = water / total_score

                # -----------------------------
                # Atmosphere templates
                # -----------------------------
                # Gas giant
                h2_r = 0.75 + 0.15 * np.random.rand()
                he_r = 0.10 + 0.05 * np.random.rand()
                n2_r = 0.005 + 0.005 * np.random.rand()

                # Venus-like
                h2_o = 0.001
                he_o = 0.001
                n2_o = 0.03 + 0.02 * np.random.rand()

                # Earth-like
                h2_w = 0.01
                he_w = 0.001
                n2_w = 0.75 + 0.05 * np.random.rand()

                # Blend
                h2_frac = r_w * h2_r + o_w * h2_o + w_w * h2_w
                he_frac = r_w * he_r + o_w * he_o + w_w * he_w
                n2_frac = r_w * n2_r + o_w * n2_o + w_w * n2_w

                # Oxygen chemistry
                o2_frac = w_w * (0.18 + 0.05 * np.random.rand())
                o3_frac = o2_frac * (1e-6 + 1e-5 * np.random.rand())

                # Sulfur chemistry
                so2_frac = o_w * (co2 * (0.001 + 0.002 * np.random.rand()))
                h2s_frac = r_w * (ch4 * (0.005 + 0.01 * np.random.rand()))

                raw = np.array([
                    h2o, co2, ch4, co, nh3,
                    h2_frac, he_frac, n2_frac,
                    o2_frac, o3_frac, so2_frac, h2s_frac
                ], dtype=np.float64)

                # 🔹 Safer noise
                noise_vec = np.random.normal(0, 0.02, size=raw.shape)
                raw = raw * (1 + noise_vec)

                raw = np.clip(raw, 1e-30, None)

                # Normalize to %
                raw = raw / raw.sum() * 100.0

                labels.append(raw.astype(np.float32))

            except Exception as e:
                print(f"Skipping planet {planet_id}: {e}")
                continue

    specs = np.array(specs, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)

    print("Before balancing:", Counter(regimes))

    # -----------------------------
    # Balance dataset
    # -----------------------------
    min_count = min(Counter(regimes).values())
    balanced_indices = []

    for r in range(3):
        idxs = [i for i, val in enumerate(regimes) if val == r]
        np.random.shuffle(idxs)
        balanced_indices.extend(idxs[:min_count])

    specs = specs[balanced_indices]
    labels = labels[balanced_indices]

    print("After balancing:", len(balanced_indices))

    # -----------------------------
    # Shuffle dataset
    # -----------------------------
    idx = np.random.permutation(len(specs))
    specs = specs[idx]
    labels = labels[idx]

    print("Final dataset:", specs.shape, labels.shape)

    # -----------------------------
    # Save dataset
    # -----------------------------
    with h5py.File(output_h5_path, 'w') as fout:
        fout.create_dataset('X', data=specs, compression='gzip')
        fout.create_dataset('y', data=labels, compression='gzip')

        fout.attrs['gases'] = 'H2O,CO2,CH4,CO,NH3,H2,He,N2,O2,O3,SO2,H2S'
        fout.attrs['feature_mean'] = float(np.mean(specs))
        fout.attrs['feature_std'] = float(np.std(specs))

    print("✓ Dataset created successfully")
    print("Saved to:", output_h5_path)

    return specs, labels


# RUN
if __name__ == '__main__':
    create_training_dataset_with_groundtruth(
        'app/data/abc_dataset.hdf5',
        'app/data/FM_Parameter_Table.csv',
        'app/data/training_data.hdf5'
    )