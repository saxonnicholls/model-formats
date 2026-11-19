#!/usr/bin/env python3
#
#  crosscheck.py
#  model-formats
#
#  Copyright 2010-2026 Saxon Herschel Nicholls
#
#  Independent readers against our files, and our readers against theirs:
#
#    ours -> theirs   mf_crosscheck writes GGUF (every value type; f32, f16, bf16, q8_0), llm.c GPT-2 as llama.cpp
#                     GGUF, safetensors of every dtype, .npz and .npy; gguf-py, safetensors and numpy read them
#    theirs -> ours   gguf-py's GGUFWriter, safetensors.serialize, np.savez / np.save, and a torch.save-shaped zip
#                     pickled by Python's own pickle (fake torch classes, protocol 2) are read by mf_crosscheck
#
#  Both sides print "file<TAB>name<TAB>type<TAB>shape<TAB>sha256" per tensor and "file<TAB>key<TAB>type<TAB>value"
#  per GGUF key; the sets must be equal. Run with -I: nothing is imported from the working directory.
#
#      python3 -I scripts/crosscheck.py --binary build/tests/mf_crosscheck --gguf-py <dir holding gguf/> --work <dir>
#

import argparse
import hashlib
import io
import os
import pickle
import subprocess
import sys
import types
import zipfile


def run_ours(binary, *args):
    r = subprocess.run([binary, *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("crosscheck: %s %s failed:\n%s" % (binary, " ".join(args), r.stderr))
    return set(line for line in r.stdout.splitlines() if "\t" in line)


def sha(b):
    return hashlib.sha256(bytes(b)).hexdigest()


def fmt_float(x):
    return "%.17g" % float(x)


def gguf_lines(path, gguf):
    import numpy as np
    name = os.path.basename(path)
    reader = gguf.GGUFReader(path)
    out = set()
    for key, field in reader.fields.items():
        if key.startswith("GGUF."):
            continue
        vt = gguf.GGUFValueType(field.types[0])
        if vt == gguf.GGUFValueType.ARRAY:
            et = gguf.GGUFValueType(field.types[1])
            n = len(field.data)
            text = "[%s x %d]" % (et.name, n)
            if et == gguf.GGUFValueType.STRING:
                text += "".join("|" + bytes(field.parts[i]).decode("utf-8") for i in field.data)
            elif et not in (gguf.GGUFValueType.FLOAT32, gguf.GGUFValueType.FLOAT64, gguf.GGUFValueType.BOOL):
                text += "".join("|" + str(int(field.parts[i][0])) for i in field.data)
        else:
            v = field.contents()
            if vt == gguf.GGUFValueType.STRING:
                text = v
            elif vt == gguf.GGUFValueType.BOOL:
                text = "true" if v else "false"
            elif vt in (gguf.GGUFValueType.FLOAT32, gguf.GGUFValueType.FLOAT64):
                text = fmt_float(v)
            else:
                text = str(int(v))
        out.add("%s\t%s\t%s\t%s" % (name, key, vt.name, text))
    for t in reader.tensors:
        shape = ",".join(str(int(d)) for d in reversed(list(t.shape)))
        out.add("%s\t%s\t%s\t%s\t%s" % (name, t.name, gguf.GGMLQuantizationType(t.tensor_type).name.lower(), shape,
                                        sha(np.ascontiguousarray(t.data).tobytes())))
    return out


NP_NAMES = {"float64": "F64", "float32": "F32", "float16": "F16", "int64": "I64", "int32": "I32", "int16": "I16", "int8": "I8",
            "uint64": "U64", "uint32": "U32", "uint16": "U16", "uint8": "U8", "bool": "BOOL"}


def numpy_lines(path, name_of_single="array"):
    import numpy as np
    name = os.path.basename(path)
    out = set()
    loaded = np.load(path, allow_pickle=False)
    items = loaded.items() if hasattr(loaded, "items") else [(name_of_single, loaded)]
    for k, a in items:
        out.add("%s\t%s\t%s\t%s\t%s" % (name, k, NP_NAMES[str(a.dtype)], ",".join(str(d) for d in a.shape), sha(a.tobytes())))
    return out


def safetensors_lines(path):
    import safetensors
    name = os.path.basename(path)
    out = set()
    with open(path, "rb") as f:
        for k, t in safetensors.deserialize(f.read()):
            out.add("%s\t%s\t%s\t%s\t%s" % (name, k, t["dtype"], ",".join(str(d) for d in t["shape"]), sha(t["data"])))
    return out


def compare(what, ours, theirs):
    if ours != theirs:
        print("crosscheck: %s DIFFER" % what)
        for line in sorted(ours - theirs):
            print("  only ours:   " + line)
        for line in sorted(theirs - ours):
            print("  only theirs: " + line)
        return False
    print("crosscheck: %-34s agree (%d lines)" % (what, len(ours)))
    return True


def fake_torch_zip(path, tensors):
    """A torch.save archive for {name: (storage type, numpy array)}, pickled by Python's pickle with stand-in classes
    named as torch's, so the opcode stream is the one torch.save emits (protocol 2, persistent ids for storages)."""
    torch = types.ModuleType("torch")
    utils = types.ModuleType("torch._utils")
    sys.modules["torch"], sys.modules["torch._utils"] = torch, utils

    def make_storage(type_name):
        cls = type(type_name, (), {"__module__": "torch"})
        setattr(torch, type_name, cls)
        return cls

    def _rebuild_tensor_v2(*args):
        raise RuntimeError("never called")
    _rebuild_tensor_v2.__module__ = "torch._utils"
    _rebuild_tensor_v2.__qualname__ = "_rebuild_tensor_v2"
    utils._rebuild_tensor_v2 = _rebuild_tensor_v2

    class Storage:
        def __init__(self, cls, key, numel):
            self.cls, self.key, self.numel = cls, key, numel

    class Tensor:
        def __init__(self, storage, shape):
            self.storage, self.shape = storage, shape

        def __reduce_ex__(self, proto):
            import collections
            stride, s = [], 1
            for d in reversed(self.shape):
                stride.insert(0, s)
                s *= d
            return (_rebuild_tensor_v2, (self.storage, 0, tuple(self.shape), tuple(stride), False, collections.OrderedDict()))

    class Pickler(pickle.Pickler):
        def persistent_id(self, obj):
            if isinstance(obj, Storage):
                return ("storage", obj.cls, obj.key, "cpu", obj.numel)
            return None

    import collections
    state, blobs = collections.OrderedDict(), {}
    for i, (name, (type_name, arr)) in enumerate(tensors.items()):
        st = Storage(make_storage(type_name), str(i), arr.size)
        state[name] = Tensor(st, list(arr.shape))
        blobs[str(i)] = arr.tobytes()
    buf = io.BytesIO()
    Pickler(buf, protocol=2).dump(state)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as z:
        z.writestr("archive/data.pkl", buf.getvalue())
        z.writestr("archive/byteorder", "little")
        for k, b in blobs.items():
            z.writestr("archive/data/" + k, b)
    for m in ("torch", "torch._utils"):
        del sys.modules[m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--gguf-py", required=True, help="a directory holding gguf-py's gguf package")
    ap.add_argument("--work", required=True)
    ap.add_argument("--llama-simple", help="llama.cpp's llama-simple, built out of tree: load our GPT-2 GGUF and generate")
    args = ap.parse_args()
    sys.path.insert(0, os.path.abspath(args.gguf_py))
    import gguf
    import numpy as np
    import safetensors

    ok = True
    ours_dir = os.path.join(args.work, "ours")
    subprocess.run([args.binary, "write", ours_dir], check=True, capture_output=True)
    files = {f: os.path.join(ours_dir, f) for f in ("ours.gguf", "gpt2.gguf", "ours.safetensors", "ours.npz", "ours.npy")}
    for f, p in files.items():
        mine = run_ours(args.binary, "read", p)
        theirs = gguf_lines(p, gguf) if f.endswith(".gguf") else safetensors_lines(p) if f.endswith(".safetensors") else numpy_lines(p)
        ok &= compare("ours, read by %s: %s" % ("gguf-py" if f.endswith(".gguf") else "safetensors" if "safet" in f else "numpy", f), mine, theirs)
    from safetensors import safe_open
    with safe_open(files["ours.safetensors"], framework="numpy") as s:   # metadata through the package's own reader
        meta = s.metadata()
    ok &= compare("ours.safetensors __metadata__", {str(sorted(meta.items()))}, {str([("format", "pt"), ("source", "model-formats")])})

    theirs_dir = os.path.join(args.work, "theirs")
    os.makedirs(theirs_dir, exist_ok=True)
    rng = np.random.default_rng(7)
    arrays = {"f64": rng.standard_normal((2, 3)), "f32": rng.standard_normal((4, 5)).astype(np.float32),
              "f16": rng.standard_normal((3,)).astype(np.float16), "i64": np.arange(6, dtype=np.int64).reshape(2, 3),
              "u8": np.arange(7, dtype=np.uint8), "b": np.array([True, False, True]), "scalar": np.array(3.5, dtype=np.float32)}
    p = os.path.join(theirs_dir, "theirs.npz")
    np.savez(p, **arrays)
    ok &= compare("np.savez, read by ours", run_ours(args.binary, "read", p), numpy_lines(p))
    p = os.path.join(theirs_dir, "theirs.npy")
    np.save(p, arrays["f32"])
    ok &= compare("np.save, read by ours", run_ours(args.binary, "read", p), numpy_lines(p))
    st = {"a": {"dtype": "F32", "shape": [4, 5], "data": arrays["f32"].tobytes()},
          "bf": {"dtype": "BF16", "shape": [2, 2], "data": bytes([0x80, 0x3f, 0x00, 0x40, 0x40, 0x40, 0x80, 0xbf])},
          "e4": {"dtype": "F8_E4M3", "shape": [3], "data": bytes([0x38, 0x40, 0xb8])},
          "e5": {"dtype": "F8_E5M2", "shape": [3], "data": bytes([0x3c, 0x40, 0xbc])},
          "i": {"dtype": "I32", "shape": [3], "data": np.array([1, -2, 3], np.int32).tobytes()},
          "bool": {"dtype": "BOOL", "shape": [2], "data": bytes([1, 0])}}
    p = os.path.join(theirs_dir, "theirs.safetensors")
    torch_names = {"F32": "float32", "BF16": "bfloat16", "F8_E4M3": "float8_e4m3fn", "F8_E5M2": "float8_e5m2", "I32": "int32", "BOOL": "bool"}
    if hasattr(safetensors, "TensorSpec"):   # safetensors 0.8: raw buffers described by TensorSpec
        buffers = {k: np.frombuffer(t["data"], dtype=np.uint8).copy() for k, t in st.items()}
        specs = {k: safetensors.TensorSpec(dtype=torch_names[t["dtype"]], shape=t["shape"], data_ptr=buffers[k].ctypes.data, data_len=len(t["data"])) for k, t in st.items()}
        data = safetensors.serialize(specs, metadata={"format": "pt"})
    else:
        data = safetensors.serialize(st, metadata={"format": "pt"})
    with open(p, "wb") as f:
        f.write(data)
    ok &= compare("safetensors.serialize, read by ours", run_ours(args.binary, "read", p), safetensors_lines(p))
    p = os.path.join(theirs_dir, "theirs.gguf")
    w = gguf.GGUFWriter(p, "crosscheck")
    w.add_uint8("t.u8", 7)
    w.add_int16("t.i16", -5)
    w.add_uint32("t.u32", 123456)
    w.add_float32("t.f32", 0.1)
    w.add_float64("t.f64", 2.5)
    w.add_bool("t.b", False)
    w.add_string("t.s", "llama")
    w.add_uint64("t.u64", 1 << 40)
    w.add_int64("t.i64", -(1 << 40))
    w.add_array("t.strs", ["x", "yy", ""])
    w.add_array("t.ints", [1, 2, 3])
    w.add_tensor("w", rng.standard_normal((8, 32)).astype(np.float32))
    w.add_tensor("h", rng.standard_normal((2, 32)).astype(np.float16))
    q = gguf.quants.quantize(rng.standard_normal((2, 64)).astype(np.float32), gguf.GGMLQuantizationType.Q8_0)
    w.add_tensor("q", q, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    ok &= compare("gguf-py GGUFWriter, read by ours", run_ours(args.binary, "read", p), gguf_lines(p, gguf))
    p = os.path.join(theirs_dir, "theirs.pt")
    fake_torch_zip(p, {"layer.weight": ("FloatStorage", arrays["f32"]), "layer.bias": ("HalfStorage", arrays["f16"]),
                       "steps": ("LongStorage", arrays["i64"])})
    expected = {"theirs.pt\t%s\t%s\t%s\t%s" % (k, t, ",".join(str(d) for d in a.shape), sha(a.tobytes()))
                for k, (t, a) in {"layer.weight": ("F32", arrays["f32"]), "layer.bias": ("F16", arrays["f16"]), "steps": ("I64", arrays["i64"])}.items()}
    ok &= compare("a torch.save-shaped zip, read by ours", run_ours(args.binary, "read", p), expected)
    if args.llama_simple:
        r = subprocess.run([args.llama_simple, "-m", files["gpt2.gguf"], "-n", "4", "ab"], capture_output=True, text=True, timeout=300)
        loaded = r.returncode == 0 and "decoded" in (r.stdout + r.stderr)
        print("crosscheck: %-34s %s" % ("llama.cpp loads and runs gpt2.gguf", "yes" if loaded else "NO:\n" + r.stderr[-2000:]))
        ok &= loaded
    print("crosscheck: %s" % ("all agree" if ok else "DIFFERENCES FOUND"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
