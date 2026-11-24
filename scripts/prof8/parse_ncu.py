import csv, sys, json, re
R = "/mnt/home/jordan+123/history-native-20261015-r5/maceoff-study/results/prof8"
rows = list(csv.reader(open(f"{R}/flashtp_full_raw.csv")))
hdr, units, data = rows[0], rows[1], rows[2:]
print(len(data), "kernels")
want = {
 "dur_us": "gpu__time_duration.sum", "grid": "Grid Size", "block": "Block Size",
 "regs": "launch__registers_per_thread", "smem_dyn_B": "launch__shared_mem_per_block_dynamic",
 "occ_theor_%": "sm__maximum_warps_per_active_cycle_pct", "occ_ach_%": "sm__warps_active.avg.pct_of_peak_sustained_active",
 "blocks_per_sm_lim_smem": "launch__occupancy_limit_shared_mem", "blocks_per_sm_lim_regs": "launch__occupancy_limit_registers",
 "waves": "launch__waves_per_multiprocessor",
 "sm_thru_%": "sm__throughput.avg.pct_of_peak_sustained_elapsed", "mem_thru_%": "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
 "dram_%": "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed", "dram_bytes": "dram__bytes.sum",
 "l2_%": "lts__t_sectors.avg.pct_of_peak_sustained_elapsed", "l1_%": "l1tex__throughput.avg.pct_of_peak_sustained_active",
 "fp64_pipe_%": "sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active",
 "dfma": "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum", "dmul": "smsp__sass_thread_inst_executed_op_dmul_pred_on.sum",
 "dadd": "smsp__sass_thread_inst_executed_op_dadd_pred_on.sum",
 "atom_red_l2_req": "lts__t_requests_srcunit_l1_op_red.sum", "atom_l2_req_atom": "lts__t_requests_srcunit_l1_op_atom.sum",
 "red_global_inst": "smsp__inst_executed_op_global_red.sum", "atom_global_inst": "smsp__inst_executed_op_global_atom.sum",
 "lts_red_%": "lts__t_sectors_srcunit_l1_op_red.avg.pct_of_peak_sustained_elapsed",
 "ipc": "sm__inst_executed.avg.per_cycle_active", "issue_%": "sm__inst_issued.avg.pct_of_peak_sustained_active",
 "elapsed_cyc": "gpu__time_duration.sum",
}
# stall reasons
stall_cols = [i for i, h in enumerate(hdr) if re.match(r"smsp__average_warp_latency_issue_stalled_.*\.ratio$", h) or re.match(r"smsp__pcsamp_warps_issue_stalled_[a-z_]+$", h)]
idx = {k: (hdr.index(v) if v in hdr else None) for k, v in want.items()}
print({k: v for k, v in idx.items() if v is None})
SYS = ["aspirin", "water192", "org192", "water648", "org648"]
out = []
for n, r in enumerate(data):
    si, rem = divmod(n, 8); layer, rem = divmod(rem, 4); it, d = divmod(rem, 2)
    if it != 1: continue
    D = dict(system=SYS[si], layer=layer, dir="fwd" if d == 0 else "bwd", kernel=r[hdr.index("Kernel Name")][:40])
    for k, i in idx.items():
        if i is not None: D[k] = r[i]
    st = {hdr[i]: r[i] for i in stall_cols}
    def f(x):
        try: return float(x.replace(",", ""))
        except: return 0.0
    st = sorted(((k.replace("smsp__pcsamp_warps_issue_stalled_", "").replace("smsp__average_warp_latency_issue_stalled_", ""), f(v)) for k, v in st.items() if "not_issued" not in k), key=lambda t: -t[1])
    tot = sum(v for k, v in st if not k.endswith(".ratio")) or 1
    D["top_stalls_pcsamp"] = [(k, round(v / tot * 100, 1)) for k, v in st if not k.endswith(".ratio")][:5]
    out.append(D)
units_d = {want[k]: units[idx[k]] for k in idx if idx[k] is not None}
json.dump(dict(units=units_d, kernels=out), open(f"{R}/flashtp_ncu_summary.json", "w"), indent=1)
for D in out: print(json.dumps(D))
print(units_d)
