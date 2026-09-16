import sys
import gguf

DRAFT, TARGET, DST = sys.argv[1], sys.argv[2], sys.argv[3]
NAMES = ["token_embd.weight", "output.weight"]

dr = gguf.GGUFReader(DRAFT)
tr = gguf.GGUFReader(TARGET)
have = {t.name for t in dr.tensors}
extra = [t for t in tr.tensors if t.name in NAMES and t.name not in have]
print("adding", [t.name for t in extra])

arch = dr.fields[gguf.Keys.General.ARCHITECTURE].contents()
w = gguf.GGUFWriter(DST, arch=arch, endianess=dr.endianess)
for f in dr.fields.values():
    if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF."):
        continue
    vt = f.types[0]
    st = f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
    w.add_key_value(f.name, f.contents(), vt, sub_type=st)

for t in list(dr.tensors) + extra:
    w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_ti_data_to_file()
for t in list(dr.tensors) + extra:
    w.write_tensor_data(t.data, tensor_endianess=dr.endianess)
w.close()
print("done")
