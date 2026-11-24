import torch, cuequivariance as cue
from e3nn import o3
from mace.tools.cg import O3_e3nn
from mace.modules.irreps_tools import tp_out_irreps_with_instructions
for xin in ["128x0e", "128x0e+128x1o"]:
    sh = o3.Irreps("1x0e+1x1o+1x2e+1x3o")
    target = o3.Irreps("128x0e+128x1o+128x2e+128x3o")
    mid, ins = tp_out_irreps_with_instructions(o3.Irreps(xin), sh, target)
    tp = o3.TensorProduct(xin, sh, mid, ins, shared_weights=False, internal_weights=False)
    print("e3nn mid", mid, "| conv irreps_out", tp.irreps_out)
    print("e3nn ins", [(i.i_in1, i.i_in2, i.i_out) for i in tp.instructions])
    d = cue.descriptors.channelwise_tensor_product(cue.Irreps(O3_e3nn, xin), cue.Irreps(O3_e3nn, str(sh)), cue.Irreps(O3_e3nn, str(mid)))
    print("cueq outputs", [str(o.irreps) for o in d.outputs], "inputs", [str(o.irreps) for o in d.inputs])
