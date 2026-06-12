# SPDX-License-Identifier: Apache-2.0
"""Python bindings for vccl (Vulkan Collective Communications Library).

Pure ctypes — no build step. Works on any object exposing a writable buffer
(numpy arrays, bytearrays, torch CPU tensors via ``data_ptr()``, or raw
integer addresses).

Example (one process per rank)::

    import vccl

    if rank == 0:
        uid = vccl.get_unique_id()       # distribute out-of-band
    comm = vccl.Communicator(nranks, uid, rank)
    comm.all_reduce(send_arr, recv_arr, op=vccl.RedOp.SUM)
    comm.destroy()
"""

from __future__ import annotations

import ctypes
import ctypes.util
import enum
import os
from typing import Any, Optional, Tuple

__all__ = [
    "DataType",
    "RedOp",
    "VcclError",
    "Communicator",
    "UniqueId",
    "get_unique_id",
    "UNIQUE_ID_BYTES",
]

UNIQUE_ID_BYTES = 128


class DataType(enum.IntEnum):
    INT8 = 0
    UINT8 = 1
    INT32 = 2
    UINT32 = 3
    INT64 = 4
    UINT64 = 5
    FLOAT16 = 6
    FLOAT32 = 7
    FLOAT64 = 8
    BFLOAT16 = 9


class RedOp(enum.IntEnum):
    SUM = 0
    PROD = 1
    MAX = 2
    MIN = 3
    AVG = 4


_TYPE_SIZE = {
    DataType.INT8: 1,
    DataType.UINT8: 1,
    DataType.INT32: 4,
    DataType.UINT32: 4,
    DataType.INT64: 8,
    DataType.UINT64: 8,
    DataType.FLOAT16: 2,
    DataType.FLOAT32: 4,
    DataType.FLOAT64: 8,
    DataType.BFLOAT16: 2,
}

# numpy dtype names / torch dtype names -> DataType
_DTYPE_NAMES = {
    "int8": DataType.INT8,
    "uint8": DataType.UINT8,
    "int32": DataType.INT32,
    "uint32": DataType.UINT32,
    "int64": DataType.INT64,
    "uint64": DataType.UINT64,
    "float16": DataType.FLOAT16,
    "half": DataType.FLOAT16,
    "float32": DataType.FLOAT32,
    "float": DataType.FLOAT32,
    "float64": DataType.FLOAT64,
    "double": DataType.FLOAT64,
    "bfloat16": DataType.BFLOAT16,
}


class VcclError(RuntimeError):
    def __init__(self, result: int, what: str):
        self.result = result
        super().__init__(f"{what}: {_lib().vcclGetErrorString(result).decode()}")


class UniqueId(ctypes.Structure):
    # c_ubyte, not c_char: ctypes truncates c_char arrays at the first NUL.
    _fields_ = [("internal", ctypes.c_ubyte * UNIQUE_ID_BYTES)]

    def hex(self) -> str:
        return bytes(self.internal).hex()

    @classmethod
    def from_bytes(cls, raw: bytes) -> "UniqueId":
        if len(raw) != UNIQUE_ID_BYTES:
            raise ValueError("bad unique id length")
        uid = cls()
        ctypes.memmove(uid.internal, raw, UNIQUE_ID_BYTES)
        return uid

    @classmethod
    def from_hex(cls, s: str) -> "UniqueId":
        return cls.from_bytes(bytes.fromhex(s.strip()))


_LIB: Optional[ctypes.CDLL] = None


def _lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    candidates = []
    env = os.environ.get("VCCL_LIBRARY")
    if env:
        candidates.append(env)
    here = os.path.dirname(os.path.abspath(__file__))
    for name in ("libvccl.so", "libvccl.dylib"):
        candidates.append(os.path.join(here, name))
        candidates.append(name)
    found = ctypes.util.find_library("vccl")
    if found:
        candidates.append(found)
    last_err: Optional[Exception] = None
    for cand in candidates:
        try:
            lib = ctypes.CDLL(cand)
        except OSError as exc:
            last_err = exc
            continue
        _setup_prototypes(lib)
        _LIB = lib
        return lib
    raise OSError(
        f"cannot load libvccl (tried {candidates}); set VCCL_LIBRARY"
    ) from last_err


def _setup_prototypes(lib: ctypes.CDLL) -> None:
    c = ctypes
    lib.vcclGetErrorString.restype = c.c_char_p
    lib.vcclGetErrorString.argtypes = [c.c_int]
    lib.vcclGetUniqueId.argtypes = [c.POINTER(UniqueId)]
    lib.vcclCommInitRank.argtypes = [
        c.POINTER(c.c_void_p), c.c_int, UniqueId, c.c_int]
    lib.vcclCommDestroy.argtypes = [c.c_void_p]
    lib.vcclCommCount.argtypes = [c.c_void_p, c.POINTER(c.c_int)]
    lib.vcclCommUserRank.argtypes = [c.c_void_p, c.POINTER(c.c_int)]
    lib.vcclAllGather.argtypes = [
        c.c_void_p, c.c_void_p, c.c_size_t, c.c_int, c.c_void_p]
    lib.vcclReduceScatter.argtypes = [
        c.c_void_p, c.c_void_p, c.c_size_t, c.c_int, c.c_int, c.c_void_p]
    lib.vcclAllReduce.argtypes = [
        c.c_void_p, c.c_void_p, c.c_size_t, c.c_int, c.c_int, c.c_void_p]
    lib.vcclBroadcast.argtypes = [
        c.c_void_p, c.c_void_p, c.c_size_t, c.c_int, c.c_int, c.c_void_p]
    lib.vcclSend.argtypes = [
        c.c_void_p, c.c_size_t, c.c_int, c.c_int, c.c_void_p]
    lib.vcclRecv.argtypes = [
        c.c_void_p, c.c_size_t, c.c_int, c.c_int, c.c_void_p]
    lib.vcclCommRegister.argtypes = [
        c.c_void_p, c.c_void_p, c.c_size_t, c.POINTER(c.c_void_p)]
    lib.vcclCommDeregister.argtypes = [c.c_void_p, c.c_void_p]
    lib.vcclCommRegisterDmaBuf.argtypes = [
        c.c_void_p, c.c_int, c.c_uint64, c.c_size_t,
        c.POINTER(c.c_void_p), c.POINTER(c.c_void_p)]


def _check(result: int, what: str) -> None:
    if result != 0:
        raise VcclError(result, what)


def _buffer_info(obj: Any, writable: bool) -> Tuple[int, int, Optional[DataType]]:
    """Return (address, byte_length, dtype-or-None) for a buffer-ish object."""
    if isinstance(obj, int):  # raw address; caller supplies count/dtype
        return obj, 0, None
    # torch tensor (CPU)
    data_ptr = getattr(obj, "data_ptr", None)
    if callable(data_ptr):
        dtype_name = str(obj.dtype).replace("torch.", "")
        nbytes = obj.numel() * obj.element_size()
        return obj.data_ptr(), nbytes, _DTYPE_NAMES.get(dtype_name)
    # numpy array
    iface = getattr(obj, "__array_interface__", None)
    if iface is not None:
        addr, ro = iface["data"]
        if writable and ro:
            raise ValueError("buffer is read-only")
        return addr, obj.nbytes, _DTYPE_NAMES.get(obj.dtype.name)
    # generic writable buffer protocol (bytearray, writable memoryview, ...)
    view = memoryview(obj)
    if view.readonly:
        raise TypeError(
            "read-only generic buffers are not supported; use numpy/torch")
    addr = ctypes.addressof((ctypes.c_char * 0).from_buffer(obj))
    return addr, view.nbytes, None


def get_unique_id() -> UniqueId:
    uid = UniqueId()
    _check(_lib().vcclGetUniqueId(ctypes.byref(uid)), "vcclGetUniqueId")
    return uid


def group_start() -> None:
    _check(_lib().vcclGroupStart(), "vcclGroupStart")


def group_end() -> None:
    _check(_lib().vcclGroupEnd(), "vcclGroupEnd")


class Communicator:
    """One rank's handle on a vccl communicator. Calls are blocking."""

    def __init__(self, nranks: int, unique_id: UniqueId, rank: int):
        self._comm = ctypes.c_void_p()
        _check(
            _lib().vcclCommInitRank(
                ctypes.byref(self._comm), nranks, unique_id, rank),
            "vcclCommInitRank",
        )
        self.rank = rank
        self.nranks = nranks

    # ── helpers ──────────────────────────────────────────────────────────
    def _resolve(self, buf: Any, count: Optional[int],
                 dtype: Optional[DataType], writable: bool,
                 count_scale: int = 1) -> Tuple[int, int, DataType]:
        addr, nbytes, inferred = _buffer_info(buf, writable)
        dt = dtype if dtype is not None else inferred
        if dt is None:
            raise ValueError("dtype could not be inferred; pass dtype=")
        n = count
        if n is None:
            if nbytes == 0:
                raise ValueError("count required when passing raw addresses")
            n = nbytes // _TYPE_SIZE[dt] // count_scale
        return addr, n, dt

    # ── collectives ──────────────────────────────────────────────────────
    def all_reduce(self, send: Any, recv: Optional[Any] = None,
                   count: Optional[int] = None,
                   dtype: Optional[DataType] = None,
                   op: RedOp = RedOp.SUM) -> None:
        if recv is None:
            recv = send  # in place
        saddr, n, dt = self._resolve(send, count, dtype, writable=False)
        raddr, _, _ = self._resolve(recv, n, dt, writable=True)
        _check(_lib().vcclAllReduce(saddr, raddr, n, dt, op, self._comm),
               "vcclAllReduce")

    def all_gather(self, send: Any, recv: Any,
                   sendcount: Optional[int] = None,
                   dtype: Optional[DataType] = None) -> None:
        saddr, n, dt = self._resolve(send, sendcount, dtype, writable=False)
        raddr, _, _ = self._resolve(recv, n, dt, writable=True)
        _check(_lib().vcclAllGather(saddr, raddr, n, dt, self._comm),
               "vcclAllGather")

    def reduce_scatter(self, send: Any, recv: Any,
                       recvcount: Optional[int] = None,
                       dtype: Optional[DataType] = None,
                       op: RedOp = RedOp.SUM) -> None:
        # recvcount inferred from the (smaller) recv buffer when not given.
        raddr, n, dt = self._resolve(recv, recvcount, dtype, writable=True)
        saddr, _, _ = self._resolve(send, n, dt, writable=False)
        _check(
            _lib().vcclReduceScatter(saddr, raddr, n, dt, op, self._comm),
            "vcclReduceScatter")

    def broadcast(self, buf: Any, root: int, count: Optional[int] = None,
                  dtype: Optional[DataType] = None) -> None:
        addr, n, dt = self._resolve(buf, count, dtype, writable=True)
        _check(_lib().vcclBroadcast(addr, addr, n, dt, root, self._comm),
               "vcclBroadcast")

    def send(self, buf: Any, peer: int, count: Optional[int] = None,
             dtype: Optional[DataType] = None) -> None:
        addr, n, dt = self._resolve(buf, count, dtype, writable=False)
        _check(_lib().vcclSend(addr, n, dt, peer, self._comm), "vcclSend")

    def recv(self, buf: Any, peer: int, count: Optional[int] = None,
             dtype: Optional[DataType] = None) -> None:
        addr, n, dt = self._resolve(buf, count, dtype, writable=True)
        _check(_lib().vcclRecv(addr, n, dt, peer, self._comm), "vcclRecv")

    # ── memory registration ──────────────────────────────────────────────
    def register(self, buf: Any, nbytes: Optional[int] = None) -> ctypes.c_void_p:
        addr, blen, _ = _buffer_info(buf, writable=True)
        handle = ctypes.c_void_p()
        _check(
            _lib().vcclCommRegister(self._comm, addr, nbytes or blen,
                                    ctypes.byref(handle)),
            "vcclCommRegister")
        return handle

    def register_dmabuf(self, fd: int, nbytes: int,
                        offset: int = 0) -> Tuple[int, ctypes.c_void_p]:
        """Returns (mapped_address, handle)."""
        ptr = ctypes.c_void_p()
        handle = ctypes.c_void_p()
        _check(
            _lib().vcclCommRegisterDmaBuf(self._comm, fd, offset, nbytes,
                                          ctypes.byref(ptr),
                                          ctypes.byref(handle)),
            "vcclCommRegisterDmaBuf")
        return ptr.value, handle

    def deregister(self, handle: ctypes.c_void_p) -> None:
        _check(_lib().vcclCommDeregister(self._comm, handle),
               "vcclCommDeregister")

    def destroy(self) -> None:
        if self._comm:
            _lib().vcclCommDestroy(self._comm)
            self._comm = ctypes.c_void_p()

    def __enter__(self) -> "Communicator":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.destroy()
