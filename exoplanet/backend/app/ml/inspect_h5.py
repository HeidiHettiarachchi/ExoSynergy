import h5py

FNAME = 'app/data/abc_dataset.hdf5'
f = h5py.File(FNAME, 'r')

# Print top-level groups (planets)
groups = list(f.keys())
print('Num groups:', len(groups))
print('First groups sample:', groups[:5])

# Inspect a single group in detail
gname = groups[0]
g = f[gname]
print('\nInspecting group:', gname)
for k in g.keys():
    d = g[k]
    print('  ', k, type(d), getattr(d, 'shape', None))
    try:
        arr = d[()]
        print('     sample:', arr.flatten()[:5])
    except Exception:
        pass

# Print group attributes, if any
print('\nAttributes for', gname, dict(g.attrs))

f.close()

# Quick scan for gas labels in first 500 groups
f = h5py.File(FNAME, 'r')
gas_keys_found = []
for gname in list(f.keys())[:500]:
    g = f[gname]
    # check dataset names
    for k in g.keys():
        if any(gas in k.upper() for gas in ('H2O','CO2','CO','CH4','NH3','COMPOSIT')):
            gas_keys_found.append((gname, k))
    # check attributes
    for ak in g.attrs.keys():
        if any(gas in ak.upper() for gas in ('H2O','CO2','CO','CH4','NH3','COMPOSIT')):
            gas_keys_found.append((gname, 'attr:'+ak))

print('\nGas-like keys found (sample):', gas_keys_found[:10])
f.close()
