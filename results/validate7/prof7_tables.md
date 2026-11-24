job 1033 node slurm-h100-206-077 torch 2.11.0+cu128

## 1. Replay wall vs GPU busy (one graph per system; replay only, 60 synced reps after 10 warmups)

| system | backend | N/E/E_cap | replay ms median [p10-p90] | copy-in+replay ms | GPU kernel sum ms (profiled replay) | busy/replay wall | span ms | kernels+copies in graph | mean kernel us | tight-cap replay ms (pad cost) |
|---|---|---|---|---|---|---|---|---|---|---|
| aspirin | flash | 21/302/384 | 1.903 [1.901-1.905] | 1.970 | 1.702 | 89% | 2.040 | 661 | 2.6 | 1.911 (-0.5%) |
| aspirin | hybrid | 21/302/384 | 1.372 [1.369-1.376] | 1.457 | 1.148 | 84% | 1.409 | 598 | 1.9 | 1.363 (+0.6%) |
| water192 | flash | 192/6634/8000 | 6.613 [6.602-6.623] | 6.701 | 6.409 | 97% | 6.713 | 668 | 9.6 | 6.372 (+3.6%) |
| water192 | hybrid | 192/6634/8000 | 2.937 [2.925-2.943] | 3.003 | 2.690 | 92% | 2.974 | 607 | 4.4 | 2.677 (+8.9%) |
| org192 | flash | 196/3670/4416 | 6.016 [6.010-6.021] | 6.103 | 5.822 | 97% | 6.130 | 669 | 8.7 | 5.906 (+1.8%) |
| org192 | hybrid | 196/3670/4416 | 2.301 [2.299-2.305] | 2.385 | 2.076 | 90% | 2.367 | 608 | 3.4 | 2.192 (+4.8%) |
| water648 | flash | 648/26550/31872 | 19.603 [19.593-19.617] | 19.700 | 19.517 | 100% | 19.819 | 688 | 28.4 | 18.722 (+4.5%) |
| water648 | hybrid | 648/26550/31872 | 7.641 [7.631-7.647] | 7.728 | 7.433 | 97% | 7.707 | 609 | 12.2 | 6.749 (+11.7%) |
| org648 | flash | 650/14856/17856 | 17.290 [17.280-17.305] | 17.385 | 17.249 | 100% | 17.561 | 688 | 25.1 | 16.798 (+2.8%) |
| org648 | hybrid | 650/14856/17856 | 5.275 [5.269-5.283] | 5.360 | 5.052 | 96% | 5.340 | 609 | 8.3 | 4.777 (+9.4%) |
| wbox648 | flash | 648/35698/42880 | 21.506 [21.494-21.517] | 21.593 | 21.433 | 100% | 21.730 | 688 | 31.2 | 20.291 (+5.6%) |
| wbox648 | hybrid | 648/35698/42880 | 9.543 [9.538-9.550] | 9.631 | 9.361 | 98% | 9.632 | 609 | 15.4 | 8.327 (+12.7%) |

## 2. GPU time by MACE op (fraction of graphed replay wall; fwd/bwd us; #kernels fwd+bwd)

### hybrid

| op group | aspirin (E_cap 384) | water192 (E_cap 8000) | org192 (E_cap 4416) | water648 (E_cap 31872) | org648 (E_cap 17856) | wbox648 (E_cap 42880) |
|---|---|---|---|---|---|---|
| FlashTP conv (fwd+bwd kernels, gather+TP+scatter) | 10.9% (61/89; 53) | 41.6% (404/817; 53) | 30.5% (240/462; 53) | 59.9% (1494/3085; 53) | 49.4% (848/1756; 53) | 63.9% (1974/4120; 53) |
| SC (cueq / e3nn) | 4.2% (34/23; 20) | 2.5% (44/30; 20) | 3.1% (43/28; 20) | 1.7% (68/65; 20) | 2.4% (68/60; 20) | 1.4% (68/63; 20) |
| embed: SH  [B] | 8.8% (36/84; 82) | 4.7% (40/97; 82) | 5.4% (36/89; 82) | 2.0% (49/101; 82) | 2.6% (41/95; 82) | 1.7% (56/108; 82) |
| embed: radial basis+cutoff  [B] | 10.1% (54/84; 76) | 5.2% (58/96; 76) | 6.4% (55/92; 76) | 2.2% (60/104; 76) | 3.0% (59/100; 76) | 1.8% (62/114; 76) |
| embed: edge vectors/lengths + e0 (prepare_graph)  [B optional] | 5.2% (23/48; 37) | 4.0% (26/91; 47) | 4.5% (26/78; 47) | 2.0% (31/122; 51) | 2.6% (30/106; 51) | 1.7% (31/133; 51) |
| embed: radial MLP (conv_tp_weights) | 9.2% (71/55; 60) | 10.3% (162/140; 59) | 9.8% (119/107; 60) | 11.9% (488/423; 59) | 10.3% (291/251; 59) | 13.4% (680/603; 59) |
| interaction linear (+1/avg)  [C] | 12.6% (71/102; 95) | 8.2% (90/151; 95) | 10.5% (90/152; 95) | 6.6% (157/345; 95) | 9.6% (158/348; 95) | 5.3% (158/344; 95) |
| skip_tp  [C] | 9.6% (61/71; 64) | 8.2% (111/131; 64) | 11.0% (122/133; 64) | 7.7% (320/270; 62) | 11.2% (323/269; 62) | 6.2% (320/271; 62) |
| linear_up | 3.0% (18/23; 23) | 1.6% (19/28; 23) | 2.1% (20/29; 23) | 0.8% (24/40; 23) | 1.2% (24/39; 23) | 0.7% (24/43; 23) |
| products linear (+sc add) | 3.1% (20/22; 23) | 1.6% (22/24; 23) | 2.0% (22/24; 23) | 0.7% (27/30; 23) | 1.1% (26/30; 23) | 0.6% (26/30; 23) |
| readouts + energy sum/scale_shift/outputs | 6.4% (45/43; 58) | 3.3% (49/48; 58) | 4.3% (50/49; 58) | 1.5% (59/56; 58) | 2.2% (59/56; 58) | 1.2% (59/55; 58) |
| unmapped/other | 0.7% (0/10; 7) | 0.4% (0/11; 7) | 0.5% (0/11; 7) | 0.2% (0/15; 7) | 0.3% (0/16; 7) | 0.2% (0/16; 7) |
| GPU idle (launch gaps inside graph) | 16.3% | 8.4% | 9.8% | 2.7% | 4.2% | 1.9% |
| padding overhead (edge-op time x (1-E/E_cap); measured = replay vs tight cap) | ~9.4% est; measured +0.6% | ~11.2% est; measured +8.9% | ~9.6% est; measured +4.8% | ~13.0% est; measured +11.7% | ~11.4% est; measured +9.4% | ~13.8% est; measured +12.7% |
| **fusion targets** | B=18.8% (+edge vec 5.2%); C=22.2% | B=9.9% (+edge vec 4.0%); C=16.5% | B=11.8% (+edge vec 4.5%); C=21.6% | B=4.1% (+edge vec 2.0%); C=14.3% | B=5.6% (+edge vec 2.6%); C=20.8% | B=3.6% (+edge vec 1.7%); C=11.5% |

### flash

| op group | aspirin (E_cap 384) | water192 (E_cap 8000) | org192 (E_cap 4416) | water648 (E_cap 31872) | org648 (E_cap 17856) | wbox648 (E_cap 42880) |
|---|---|---|---|---|---|---|
| FlashTP conv (fwd+bwd kernels, gather+TP+scatter) | 8.2% (62/93; 53) | 18.6% (404/823; 53) | 11.8% (242/465; 53) | 23.4% (1491/3087; 53) | 15.2% (855/1767; 53) | 28.4% (1976/4121; 53) |
| SC (cueq / e3nn) | 31.2% (294/300; 82) | 57.1% (1734/2043; 81) | 63.3% (1747/2062; 81) | 62.2% (5681/6516; 99) | 71.1% (5733/6557; 99) | 56.7% (5684/6516; 99) |
| embed: SH  [B] | 6.2% (33/84; 82) | 2.1% (39/98; 82) | 2.1% (36/90; 82) | 0.8% (49/100; 82) | 0.8% (42/95; 82) | 0.8% (55/108; 82) |
| embed: radial basis+cutoff  [B] | 7.1% (51/84; 76) | 2.4% (58/98; 76) | 2.4% (55/92; 76) | 0.8% (61/106; 76) | 0.9% (60/101; 76) | 0.8% (64/113; 76) |
| embed: edge vectors/lengths + e0 (prepare_graph)  [B optional] | 3.9% (26/48; 38) | 1.8% (26/94; 47) | 1.8% (27/80; 47) | 0.8% (31/124; 51) | 0.8% (31/106; 51) | 0.8% (31/133; 51) |
| embed: radial MLP (conv_tp_weights) | 6.8% (72/57; 60) | 4.6% (164/142; 59) | 3.8% (119/109; 60) | 4.7% (493/424; 59) | 3.2% (290/262; 59) | 5.9% (677/602; 59) |
| interaction linear (+1/avg)  [C] | 9.2% (71/103; 95) | 3.7% (92/152; 95) | 4.0% (90/152; 95) | 2.6% (160/348; 95) | 2.9% (158/349; 95) | 2.4% (158/347; 95) |
| skip_tp  [C] | 7.1% (62/74; 64) | 3.6% (112/128; 64) | 4.2% (122/130; 64) | 3.0% (322/272; 62) | 3.4% (323/270; 62) | 2.8% (322/272; 62) |
| linear_up | 2.2% (18/25; 23) | 0.7% (19/29; 23) | 0.8% (20/28; 23) | 0.3% (24/41; 23) | 0.4% (24/39; 23) | 0.3% (24/43; 23) |
| products linear (+sc add) | 2.3% (23/22; 23) | 0.7% (22/24; 23) | 0.8% (22/24; 23) | 0.3% (26/30; 23) | 0.3% (26/30; 23) | 0.3% (27/30; 23) |
| readouts + energy sum/scale_shift/outputs | 4.7% (46/43; 58) | 1.4% (48/48; 58) | 1.6% (48/49; 58) | 0.6% (60/55; 58) | 0.7% (60/56; 58) | 0.5% (60/55; 58) |
| unmapped/other | 0.5% (0/10; 7) | 0.2% (0/11; 7) | 0.2% (0/11; 7) | 0.1% (0/16; 7) | 0.1% (0/16; 7) | 0.1% (0/16; 7) |
| GPU idle (launch gaps inside graph) | 10.5% | 3.1% | 3.2% | 0.4% | 0.2% | 0.3% |
| padding overhead (edge-op time x (1-E/E_cap); measured = replay vs tight cap) | ~6.9% est; measured -0.5% | ~5.0% est; measured +3.6% | ~3.7% est; measured +1.8% | ~5.1% est; measured +4.5% | ~3.5% est; measured +2.8% | ~6.1% est; measured +5.6% |
| **fusion targets** | B=13.3% (+edge vec 3.9%); C=16.3% | B=4.4% (+edge vec 1.8%); C=7.3% | B=4.5% (+edge vec 1.8%); C=8.2% | B=1.6% (+edge vec 0.8%); C=5.6% | B=1.7% (+edge vec 0.8%); C=6.4% | B=1.6% (+edge vec 0.8%); C=5.1% |

## 3. GPU time by kernel class (fraction of replay kernel sum)

### hybrid (top-25 kernel names classified; remainder = 'not in top 25')

| class | aspirin | water192 | org192 | water648 | org648 | wbox648 |
|---|---|---|---|---|---|---|
| FlashTP | 8.0% (91 us) | 42.7% (1147 us) | 30.4% (632 us) | 60.2% (4474 us) | 49.7% (2513 us) | 63.9% (5981 us) |
| cueq | 1.8% (21 us) | 0.7% (19 us) | 1.3% (27 us) | 0.6% (43 us) | 0.8% (43 us) | 0.4% (42 us) |
| GEMM(fp64) | 18.0% (207 us) | 15.6% (420 us) | 18.3% (380 us) | 15.1% (1124 us) | 17.8% (897 us) | 14.3% (1340 us) |
| elementwise | 55.3% (635 us) | 30.0% (808 us) | 37.2% (772 us) | 17.9% (1333 us) | 23.5% (1185 us) | 16.1% (1511 us) |
| copy/cat/memset | 5.6% (65 us) | 3.0% (81 us) | 4.2% (87 us) | 1.6% (120 us) | 2.2% (114 us) | 1.3% (126 us) |
| index/scatter/sort | 3.9% (45 us) | 2.5% (66 us) | 2.6% (54 us) | 1.3% (100 us) | 1.6% (82 us) | 1.2% (108 us) |
| reduce | 3.0% (35 us) | 1.8% (49 us) | 2.1% (44 us) | 1.0% (74 us) | 1.1% (58 us) | 0.9% (85 us) |
| other | 0.7% (8 us) | 0.0% (0 us) | 0.0% (0 us) | 0.0% (0 us) | 0.0% (0 us) | 0.0% (0 us) |
| not in top 25 | 3.6% (42 us) | 3.7% (99 us) | 3.8% (80 us) | 2.2% (165 us) | 3.2% (161 us) | 1.8% (168 us) |

### flash (top-25 kernel names classified; remainder = 'not in top 25')

| class | aspirin | water192 | org192 | water648 | org648 | wbox648 |
|---|---|---|---|---|---|---|
| FlashTP | 5.6% (94 us) | 18.0% (1154 us) | 10.9% (635 us) | 22.9% (4471 us) | 14.7% (2528 us) | 27.9% (5984 us) |
| cueq | 0.0% (0 us) | 0.0% (0 us) | 0.0% (0 us) | 0.0% (0 us) | 0.0% (0 us) | 0.0% (0 us) |
| GEMM(fp64) | 31.3% (532 us) | 39.7% (2546 us) | 43.2% (2516 us) | 40.8% (7963 us) | 45.4% (7825 us) | 38.1% (8174 us) |
| elementwise | 44.5% (757 us) | 29.3% (1877 us) | 31.8% (1854 us) | 25.0% (4876 us) | 27.5% (4736 us) | 23.5% (5044 us) |
| copy/cat/memset | 4.1% (70 us) | 0.8% (51 us) | 0.9% (53 us) | 0.7% (128 us) | 0.7% (120 us) | 0.6% (131 us) |
| index/scatter/sort | 2.5% (43 us) | 0.9% (54 us) | 0.7% (42 us) | 0.2% (42 us) | 0.2% (31 us) | 0.2% (47 us) |
| reduce | 3.5% (59 us) | 2.8% (178 us) | 3.1% (180 us) | 2.4% (475 us) | 2.7% (458 us) | 2.3% (485 us) |
| other | 5.3% (90 us) | 6.1% (392 us) | 6.9% (400 us) | 6.5% (1271 us) | 7.4% (1277 us) | 5.9% (1270 us) |
| not in top 25 | 3.3% (56 us) | 2.4% (156 us) | 2.4% (142 us) | 1.5% (291 us) | 1.6% (274 us) | 1.4% (299 us) |

## 4. Roofline-style estimates (min bytes = module inputs+outputs fwd, x3 for fwd+bwd; FLOPs = 2*MACs fwd, x3 fwd+bwd)

### hybrid aspirin

| op group | GPU us (fwd+bwd) | min MB moved | achieved GB/s (% of 3.35 TB/s) | GFLOP | achieved TFLOP/s (% of 67 tensor) |
|---|---|---|---|---|---|
| FlashTP conv (fwd+bwd kernels, gather+TP+scatter) | 150 | 20.8 | 139 (4.1%) | - | - |
| SC (cueq / e3nn) | 57 | 2.4 | 42 (1.3%) | - | - |
| embed: SH  [B] | 120 | 0.2 | 1 (0.0%) | - | - |
| embed: radial basis+cutoff  [B] | 138 | 0.1 | 1 (0.0%) | - | - |
| embed: radial MLP (conv_tp_weights) | 126 | 16.7 | 132 (3.9%) | 0.30 | 2.41 (3.6%) |
| interaction linear (+1/avg)  [C] | 173 | 5.7 | 33 (1.0%) | 0.12 | 0.67 (1.0%) |
| skip_tp  [C] | 132 | 2.4 | 18 (0.5%) | - | - |
| linear_up | 41 | 0.6 | 16 (0.5%) | 0.01 | 0.25 (0.4%) |
| products linear (+sc add) | 42 | 0.6 | 15 (0.5%) | 0.01 | 0.24 (0.4%) |

### hybrid water192

| op group | GPU us (fwd+bwd) | min MB moved | achieved GB/s (% of 3.35 TB/s) | GFLOP | achieved TFLOP/s (% of 67 tensor) |
|---|---|---|---|---|---|
| FlashTP conv (fwd+bwd kernels, gather+TP+scatter) | 1221 | 387.0 | 317 (9.5%) | - | - |
| SC (cueq / e3nn) | 73 | 21.9 | 300 (8.9%) | - | - |
| embed: SH  [B] | 137 | 3.6 | 27 (0.8%) | - | - |
| embed: radial basis+cutoff  [B] | 154 | 2.2 | 14 (0.4%) | - | - |
| embed: radial MLP (conv_tp_weights) | 302 | 347.1 | 1149 (34.3%) | 6.34 | 20.99 (31.3%) |
| interaction linear (+1/avg)  [C] | 241 | 51.9 | 215 (6.4%) | 1.06 | 4.38 (6.5%) |
| skip_tp  [C] | 242 | 21.9 | 91 (2.7%) | - | - |
| linear_up | 48 | 5.9 | 124 (3.7%) | 0.09 | 1.98 (3.0%) |
| products linear (+sc add) | 46 | 5.9 | 127 (3.8%) | 0.09 | 2.03 (3.0%) |

### hybrid water648

| op group | GPU us (fwd+bwd) | min MB moved | achieved GB/s (% of 3.35 TB/s) | GFLOP | achieved TFLOP/s (% of 67 tensor) |
|---|---|---|---|---|---|
| FlashTP conv (fwd+bwd kernels, gather+TP+scatter) | 4579 | 1519.7 | 332 (9.9%) | - | - |
| SC (cueq / e3nn) | 132 | 74.0 | 559 (16.7%) | - | - |
| embed: SH  [B] | 149 | 14.5 | 97 (2.9%) | - | - |
| embed: radial basis+cutoff  [B] | 164 | 8.6 | 52 (1.6%) | - | - |
| embed: radial MLP (conv_tp_weights) | 912 | 1383.0 | 1517 (45.3%) | 25.26 | 27.71 (41.4%) |
| interaction linear (+1/avg)  [C] | 502 | 175.2 | 349 (10.4%) | 3.57 | 7.10 (10.6%) |
| skip_tp  [C] | 590 | 74.0 | 125 (3.7%) | - | - |
| linear_up | 64 | 19.9 | 312 (9.3%) | 0.32 | 4.99 (7.4%) |
| products linear (+sc add) | 57 | 19.9 | 348 (10.4%) | 0.32 | 5.58 (8.3%) |

### hybrid wbox648

| op group | GPU us (fwd+bwd) | min MB moved | achieved GB/s (% of 3.35 TB/s) | GFLOP | achieved TFLOP/s (% of 67 tensor) |
|---|---|---|---|---|---|
| FlashTP conv (fwd+bwd kernels, gather+TP+scatter) | 6094 | 2002.7 | 329 (9.8%) | - | - |
| SC (cueq / e3nn) | 131 | 74.0 | 563 (16.8%) | - | - |
| embed: SH  [B] | 165 | 19.6 | 119 (3.5%) | - | - |
| embed: radial basis+cutoff  [B] | 176 | 11.5 | 65 (1.9%) | - | - |
| embed: radial MLP (conv_tp_weights) | 1283 | 1860.6 | 1450 (43.3%) | 33.99 | 26.49 (39.5%) |
| interaction linear (+1/avg)  [C] | 502 | 175.2 | 349 (10.4%) | 3.57 | 7.10 (10.6%) |
| skip_tp  [C] | 591 | 74.0 | 125 (3.7%) | - | - |
| linear_up | 67 | 19.9 | 296 (8.8%) | 0.32 | 4.74 (7.1%) |
| products linear (+sc add) | 56 | 19.9 | 354 (10.6%) | 0.32 | 5.66 (8.5%) |

