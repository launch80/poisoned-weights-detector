"""Pickle opcode scanner (stdlib pickletools only). Never unpickles anything.

Handles raw pickles and torch zip checkpoints (every *.pkl member). Reports every GLOBAL /
STACK_GLOBAL target and classifies it as allowlisted, dangerous, or unknown.
"""

from __future__ import annotations

import io
import pickletools
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

DANGEROUS_MODULES = {
    "os",
    "posix",
    "nt",
    "subprocess",
    "sys",
    "builtins",
    "__builtin__",
    "importlib",
    "imp",
    "socket",
    "requests",
    "urllib",
    "urllib2",
    "urllib3",
    "http",
    "httplib",
    "shutil",
    "pty",
    "code",
    "codeop",
    "runpy",
    "webbrowser",
    "ctypes",
    "pickle",
    "_pickle",
    "cPickle",
    "base64",
    "marshal",
    "types",
    "operator",
    "functools",
    "commands",
    "timeit",
    "platform",
    "tempfile",
    "multiprocessing",
    "threading",
    "asyncio",
    "smtplib",
    "ftplib",
    "telnetlib",
    "paramiko",
    "torch.hub",
    "pip",
    "setuptools",
    "distutils",
    "zipimport",
    "pydoc",
    "inspect",
    "gc",
    "signal",
    "resource",
    "select",
    "ssl",
    "xmlrpc",
    "aiohttp",
    "numpy.testing",
    "bdb",
    "pdb",
    "shelve",
    "dbm",
    "sqlite3",
    "glob",
    "fnmatch",
    "pathlib",
    "io",
    "codecs",
    "linecache",
}
DANGEROUS_NAMES = {
    "eval",
    "exec",
    "compile",
    "open",
    "system",
    "popen",
    "popen2",
    "spawn",
    "spawnl",
    "getattr",
    "setattr",
    "__import__",
    "import_module",
    "load",
    "loads",
    "run",
    "call",
    "check_output",
    "check_call",
    "Popen",
    "execv",
    "execve",
    "execl",
    "fork",
    "kill",
    "remove",
    "unlink",
    "rmtree",
    "urlopen",
    "urlretrieve",
    "socket",
    "connect",
    "b64decode",
    "decodebytes",
    "apply",
    "breakpoint",
    "input",
    "memoryview",
    "read",
    "write",
    "listdir",
    "walk",
    "getenv",
    "putenv",
    "environ",
    "chmod",
    "chown",
    "symlink",
    "mkfifo",
    "WinDLL",
    "CDLL",
    "cdll",
}
ALLOWLIST_MODULES = {
    "collections",
    "torch",
    "torch._utils",
    "torch._tensor",
    "torch.nn.parameter",
    "torch.serialization",
    "torch.storage",
    "numpy",
    "numpy.core.multiarray",
    "numpy._core.multiarray",
    "_codecs",
    "torch.distributed._shard",
    "typing",
    "numpy.dtypes",
    "numpy._core",
    "numpy.core",
}
ALLOWLIST = {
    ("collections", "OrderedDict"),
    ("collections", "defaultdict"),
    ("torch._utils", "_rebuild_tensor_v2"),
    ("torch._utils", "_rebuild_tensor_v3"),
    ("torch._utils", "_rebuild_parameter"),
    ("torch._utils", "_rebuild_parameter_with_state"),
    ("torch._utils", "_rebuild_meta_tensor_no_storage"),
    ("torch._utils", "_rebuild_qtensor"),
    ("torch._utils", "_rebuild_sparse_tensor"),
    ("torch._utils", "_rebuild_wrapper_subclass"),
    ("torch._tensor", "_rebuild_from_type_v2"),
    ("torch.nn.parameter", "Parameter"),
    ("torch", "Size"),
    ("torch", "device"),
    ("torch", "dtype"),
    ("torch.serialization", "_get_layout"),
    ("torch.storage", "_load_from_bytes"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "scalar"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("numpy.dtypes", "Float32DType"),
    ("_codecs", "encode"),
}
STORAGE_SUFFIXES = (
    "Storage",
)  # torch.FloatStorage, torch.BFloat16Storage, ... (torch.UntypedStorage too)
BUILD_OPS = {"REDUCE", "INST", "OBJ", "NEWOBJ", "NEWOBJ_EX", "BUILD"}


@dataclass
class GlobalRef:
    module: str
    name: str
    member: str  # zip member or "<raw>"
    pos: int
    klass: str  # allowlisted | dangerous | unknown

    @property
    def qualname(self) -> str:
        return f"{self.module}.{self.name}"


@dataclass
class PickleScan:
    path: str
    members: list[str] = field(default_factory=list)
    globals: list[GlobalRef] = field(default_factory=list)
    build_ops: dict[str, int] = field(default_factory=dict)
    protocol: int | None = None
    errors: list[str] = field(default_factory=list)
    trailing_bytes: int = 0

    @property
    def dangerous(self) -> list[GlobalRef]:
        return [g for g in self.globals if g.klass == "dangerous"]

    @property
    def unknown(self) -> list[GlobalRef]:
        return [g for g in self.globals if g.klass == "unknown"]


def classify(module: str, name: str) -> str:
    if (module, name) in ALLOWLIST:
        return "allowlisted"
    if module == "torch" and name.endswith(STORAGE_SUFFIXES):
        return "allowlisted"
    top = module.split(".")[0]
    if module in DANGEROUS_MODULES or top in DANGEROUS_MODULES or name in DANGEROUS_NAMES:
        return "dangerous"
    return "unknown"


def _scan_stream(data: bytes, member: str, out: PickleScan) -> None:
    stream = io.BytesIO(data)
    pushed: list[str] = []  # recent string pushes, for STACK_GLOBAL
    last_pos = 0
    try:
        for op, arg, pos in pickletools.genops(stream):
            last_pos = pos
            n = op.name
            if n == "PROTO":
                out.protocol = int(arg) if out.protocol is None else out.protocol
            if n in (
                "SHORT_BINUNICODE",
                "BINUNICODE",
                "UNICODE",
                "BINUNICODE8",
                "STRING",
                "SHORT_BINSTRING",
                "BINSTRING",
            ):
                pushed.append(str(arg))
                pushed = pushed[-4:]
            elif n == "GLOBAL":
                mod, _, name = str(arg).partition(" ")
                out.globals.append(GlobalRef(mod, name, member, pos, classify(mod, name)))
            elif n == "STACK_GLOBAL":
                mod, name = (pushed[-2], pushed[-1]) if len(pushed) >= 2 else ("?", "?")
                out.globals.append(GlobalRef(mod, name, member, pos, classify(mod, name)))
            elif n in BUILD_OPS:
                out.build_ops[n] = out.build_ops.get(n, 0) + 1
            elif n == "STOP":
                rest = len(data) - stream.tell()
                if rest > 0:
                    out.trailing_bytes += rest
                    # a second pickle may follow (torch legacy format has several); keep scanning
                    try:
                        for op2, arg2, pos2 in pickletools.genops(stream):
                            if op2.name == "GLOBAL":
                                mod, _, name = str(arg2).partition(" ")
                                out.globals.append(
                                    GlobalRef(mod, name, member, pos2, classify(mod, name))
                                )
                            elif op2.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE"):
                                pushed.append(str(arg2))
                                pushed = pushed[-4:]
                            elif op2.name == "STACK_GLOBAL":
                                mod, name = (
                                    (pushed[-2], pushed[-1]) if len(pushed) >= 2 else ("?", "?")
                                )
                                out.globals.append(
                                    GlobalRef(mod, name, member, pos2, classify(mod, name))
                                )
                            elif op2.name in BUILD_OPS:
                                out.build_ops[op2.name] = out.build_ops.get(op2.name, 0) + 1
                    except Exception:
                        pass  # trailing non-pickle data (torch legacy storages) is normal
                break
    except Exception as e:  # malformed pickle is itself a finding
        out.errors.append(f"{member}@{last_pos}: {type(e).__name__}: {e}")


def scan_file(path: str | Path, max_bytes: int = 256 << 20) -> PickleScan:
    p = Path(path)
    out = PickleScan(path=str(p))
    if zipfile.is_zipfile(p):
        with zipfile.ZipFile(p) as z:
            for info in z.infolist():
                nm = info.filename
                if (
                    nm.endswith((".pkl", ".pickle"))
                    or nm.endswith("/data.pkl")
                    or nm.split("/")[-1] in ("data.pkl", "archive")
                ):
                    out.members.append(nm)
                    if info.file_size > max_bytes:
                        out.errors.append(f"{nm}: {info.file_size} bytes, over scan limit")
                        continue
                    _scan_stream(z.read(info), nm, out)
                elif nm.endswith(".py") or nm.endswith(".so"):
                    out.errors.append(f"{nm}: code member inside checkpoint archive")
        if not out.members:
            out.errors.append("zip archive without a pickle member (not a torch checkpoint?)")
        return out
    size = p.stat().st_size
    with open(p, "rb") as f:
        head = f.read(2)
        f.seek(0)
        if head[:1] != b"\x80" and head not in (
            b"(d",
            b"(l",
            b"}q",
            b"]q",
            b"cc",
            b"c_",
            b"ct",
            b"cn",
        ):
            out.errors.append("does not start like a pickle (no PROTO opcode); treated as opaque")
            return out
        data = f.read(min(size, max_bytes))
    _scan_stream(data, "<raw>", out)
    if size > max_bytes:
        out.errors.append(f"only first {max_bytes} of {size} bytes scanned")
    return out
