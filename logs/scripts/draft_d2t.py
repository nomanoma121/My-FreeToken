import sys
import numpy as np
import gguf

SRC, DST, FREQ, K = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
N_SPECIAL_START = 248044

r = gguf.GGUFReader(SRC)
out = next(t for t in r.tensors if t.name == "output.weight")
n_vocab = int(out.shape[1])
rows = np.asarray(out.data).reshape(n_vocab, -1)

freq = np.load(FREQ)
order = np.argsort(-freq, kind="stable")
sel = np.zeros(n_vocab, bool)
sel[N_SPECIAL_START:] = True
for i in order:
    if sel.sum() >= K:
        break
    sel[i] = True
ids = np.nonzero(sel)[0].astype(np.int64)
sub = np.ascontiguousarray(rows[ids])
print("draft vocab", len(ids))

arch = r.fields[gguf.Keys.General.ARCHITECTURE].contents()
w = gguf.GGUFWriter(DST, arch=arch, endianess=r.endianess)
for f in r.fields.values():
    if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF."):
        continue
    vt = f.types[0]
    st = f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
    w.add_key_value(f.name, f.contents(), vt, sub_type=st)

tensors = []
for t in r.tensors:
    if t.name == "output.weight":
        tensors.append((t.name, sub, gguf.GGMLQuantizationType.Q4_0))
    else:
        tensors.append((t.name, t.data, t.tensor_type))
tensors.append(("d2t", ids, gguf.GGMLQuantizationType.I64))

for name, data, qt in tensors:
    w.add_tensor_info(name, data.shape, data.dtype, data.nbytes, qt)
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_ti_data_to_file()
for name, data, qt in tensors:
    w.write_tensor_data(data, tensor_endianess=r.endianess)
w.close()
print("done")
