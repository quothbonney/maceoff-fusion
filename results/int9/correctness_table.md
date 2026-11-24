test_full_a: job 1144 ok=True failures=0
test_full_water2187: job 1144 ok=True failures=0
test_full_wbox2187: job 1144 ok=True failures=0
test_full_water5184: job 1144 ok=True failures=0
test_full_wbox5184: job 1144 ok=True failures=0
test_binovf: job 1143 ok=True failures=0

| test | pass | samples | worst dE/atom eV | worst max dF eV/A | worst dsigma eV/A^3 | systems | edge sets == stock (bitwise) |
|---|---|---|---|---|---|---|---|
| bin_overflow_recapture | True | 4 | 0.0e+00 | 4.4e-15 | 0.0e+00 | water648, wbox648 | - |
| fallback_paths | True | 15 | 1.5e-13 | 7.1e-15 | 5.6e-17 | water192_mixed_pbc, wbox192 | - |
| zero_edge_tiny | True | 3 | 0.0e+00 | 1.0e-15 | 0.0e+00 | H2O, farO2, zeroC | - |
| species_order_atom_count_traps | True | 7 | 1.7e-13 | 8.5e-13 | 0.0e+00 | aspirin/rev/water192 interleaved | - |
| overflow_recapture | True | 2 | 1.5e-13 | 4.9e-15 | 0.0e+00 | water192, wbox648 | - |
| overflow_forces_checked | True | 2 | 1.5e-13 | 4.9e-15 | 0.0e+00 | water192, wbox648 | - |
| overflow_compression_walk | True | 12 | 1.5e-13 | 3.6e-13 | 0.0e+00 | water192, wbox648 | - |
| systems_walk_vs_stock | True | 98 | 1.1e-12 | 2.1e-14 | 7.0e-17 | aspirin, org648, water192, water2187, water5184, water648, wbox192, wbox2187, wbox5184, wbox648 | all |
| systems_walk_vs_nodehybrid_proto | True | 98 | 6.3e-13 | 0.0e+00 | 1.3e-17 | aspirin, org648, water192, water2187, water5184, water648, wbox192, wbox2187, wbox5184, wbox648 | - |
| repeat_same_coords_tol | True | 14 | 1.8e-13 | 0.0e+00 | 0.0e+00 | aspirin, org648, water192, water2187, water5184, water648, wbox192, wbox2187, wbox5184, wbox648 | - |
