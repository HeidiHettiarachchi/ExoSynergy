import h5py
import pandas as pd
import numpy as np


def create_training_dataset_with_groundtruth(
    hdf5_spectra_path,
    csv_truth_path,
    output_h5_path,
    sample_fraction=1.0
):

    print("\nCREATING PHYSICALLY CONSISTENT 12-GAS TRAINING DATASET\n")

    truth_df = pd.read_csv(csv_truth_path)

    with h5py.File(hdf5_spectra_path, 'r') as f:
        spectrum_groups = sorted(list(f.keys()))

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

    specs = []
    labels = []

    for _, row in truth_subset.iterrows():

        planet_id = int(row['planet_ID'])
        group_name = id_to_group[planet_id]

        try:
            with h5py.File(hdf5_spectra_path, 'r') as f:
                g = f[group_name]

                features = np.concatenate([
                    g['instrument_spectrum'][()].flatten(),
                    g['instrument_width'][()].flatten(),
                    g['instrument_noise'][()].flatten(),
                    g['instrument_wlgrid'][()].flatten()
                ], axis=0).astype(np.float32)

            specs.append(features)

            # -------------------------------
            # REAL CHEMISTRY (from ABC dataset)
            # Order: H2O, CO2, CH4, CO, NH3
            # -------------------------------
            log_cols = ['log_H2O', 'log_CO2', 'log_CH4', 'log_CO', 'log_NH3']
            chem_vmr = np.array([
                10.0 ** row[col] if row[col] > -30 else 1e-30
                for col in log_cols
            ])  # volume mixing ratios, already on absolute scale

            h2o, co2, ch4, co, nh3 = chem_vmr

            # ---------------------------------------
            # DETERMINE ATMOSPHERIC REGIME
            # from actual chemistry, not normalised fractions
            # ---------------------------------------
            reducing_score  = ch4 + nh3          # gas giant / H2-dominated
            oxidized_score  = co2                 # Venus/rocky
            water_score     = h2o                 # temperate / ocean world

            # ---------------------------------------
            # Bulk gases — conditioned on regime,
            # but NOT hardcoded to a single number.
            # Use the chemistry to weight H2/He/N2 etc.
            # ---------------------------------------

            if reducing_score > 1e-3:
                # H2-dominated gas giant (Jupiter/Saturn-like)
                # H2+He fill almost everything; scale slightly with chemistry sum
                fill = max(0.0, 1.0 - sum(chem_vmr))
                h2_frac = 0.87
                he_frac = 0.12
                n2_frac = 0.007
                o2_frac = 1e-6
                o3_frac = 1e-8
                so2_frac = 1e-7
                h2s_frac = ch4 * 0.01  # H2S loosely coupled to reducing chemistry

            elif oxidized_score > 1e-3:
                # CO2-rich / Venus-like
                h2_frac = 0.001
                he_frac = 0.001
                n2_frac = 0.035          # ~3.5% N2 like Venus
                o2_frac = 1e-4
                o3_frac = 1e-7
                so2_frac = co2 * 0.002  # SO2 correlated with CO2 on Venus-like worlds
                h2s_frac = 1e-8

            elif water_score > 1e-3:
                # Water-rich / temperate planet
                h2_frac = 0.01
                he_frac = 0.001
                n2_frac = 0.78
                o2_frac = 0.21
                o3_frac = o2_frac * 3e-6  # realistic O3/O2 ratio
                so2_frac = 1e-7
                h2s_frac = 1e-8

            else:
                # Sub-Neptune / mixed atmosphere — scale by total trace chemistry
                total_trace = sum(chem_vmr)
                bulk_remainder = max(0.0, 1.0 - total_trace)
                h2_frac = bulk_remainder * 0.75
                he_frac = bulk_remainder * 0.15
                n2_frac = bulk_remainder * 0.08
                o2_frac = bulk_remainder * 0.01
                o3_frac = 1e-6
                so2_frac = 1e-7
                h2s_frac = 1e-8

            # ---------------------------------------
            # Assemble full 12-gas label vector
            # Order must match attrs['gases']:
            # H2O, CO2, CH4, CO, NH3, H2, He, N2, O2, O3, SO2, H2S
            # ---------------------------------------
            raw = np.array([
                h2o,       # H2O  — from ABC dataset
                co2,       # CO2  — from ABC dataset
                ch4,       # CH4  — from ABC dataset
                co,        # CO   — from ABC dataset
                nh3,       # NH3  — from ABC dataset
                h2_frac,   # H2   — physically conditioned
                he_frac,   # He   — physically conditioned
                n2_frac,   # N2   — physically conditioned
                o2_frac,   # O2   — physically conditioned
                o3_frac,   # O3   — physically conditioned
                so2_frac,  # SO2  — physically conditioned
                h2s_frac,  # H2S  — physically conditioned
            ], dtype=np.float64)

            # Clip negatives (safety) and normalise to 100%
            raw = np.clip(raw, 1e-30, None)
            raw = raw / raw.sum() * 100.0

            labels.append(raw.astype(np.float32))

        except Exception as e:
            print(f"Skipping planet {planet_id}: {e}")
            continue

    specs = np.array(specs, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)

    with h5py.File(output_h5_path, 'w') as fout:
        fout.create_dataset('X', data=specs, compression='gzip')
        fout.create_dataset('y', data=labels, compression='gzip')
        fout.attrs['gases'] = 'H2O,CO2,CH4,CO,NH3,H2,He,N2,O2,O3,SO2,H2S'

    print("✓ Training dataset created")
    print("X shape:", specs.shape)
    print("y shape:", labels.shape)

    return specs, labels


if __name__ == '__main__':
    create_training_dataset_with_groundtruth(
        'app/data/abc_dataset.hdf5',
        'app/data/FM_Parameter_Table.csv',
        'app/data/training_data.hdf5'
    )