# SPDX-License-Identifier: Apache-2.0
"""vLLM DeviceCommunicator backed by vccl.

Drop-in communicator for vLLM platforms whose tensors live in host-accessible
memory (vllm-vulkan keeps PyTorch tensors on CPU). Collectives run over
vccl's RDMA (RoCE v2) or TCP transport instead of gloo.

Wiring into vllm-vulkan (platform.py):

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "vccl.torch_comm.VcclCommunicator"

and stop forcing ``distributed_executor_backend = "uni"`` when world_size > 1
(keep ``dist_backend = "gloo"``: torch.distributed still bootstraps process
groups; vccl replaces the data-plane collectives).

The module also works without vLLM installed — `VcclCommunicator` then
derives from a minimal standalone base, which is how the unit tests run.

Pre-registration pattern (H1): decode collective *shapes* are stable even
though the caching-allocator activation tensor's address is not, so this
communicator keeps a persistent, `vcclCommRegister`-ed staging buffer per
`(role, shape, dtype)` and runs each collective on it (copying the volatile
activation in/out). This is what removes the per-token `ibv_reg_mr` from the
RDMA hot path. Non-Python consumers that call the C collectives directly
(e.g. the vllm-vulkan Rust FFI via `model.set_collective_comm(comm.handle)`)
should follow the same recipe: allocate one fixed scratch buffer per decode
shape, call `vcclCommRegister(comm, ptr, bytes, &handle)` once at setup, and
copy activations through that buffer instead of registering per call.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

import vccl

try:
    from vllm.distributed.device_communicators.base_device_communicator import (
        DeviceCommunicatorBase,
    )
except ImportError:

    class DeviceCommunicatorBase:  # type: ignore[no-redef]
        """Standalone stand-in mirroring vLLM's attribute contract."""

        def __init__(
            self,
            cpu_group: ProcessGroup,
            device: Optional[torch.device] = None,
            device_group: Optional[ProcessGroup] = None,
            unique_name: str = "",
            **_: object,
        ) -> None:
            self.cpu_group = cpu_group
            self.device = device if device is not None else torch.device("cpu")
            self.device_group = device_group
            self.unique_name = unique_name
            self.rank = dist.get_rank(cpu_group)
            self.world_size = dist.get_world_size(cpu_group)
            self.ranks = dist.get_process_group_ranks(cpu_group)
            self.rank_in_group = self.rank


_TORCH_DTYPE = {
    torch.int8: vccl.DataType.INT8,
    torch.uint8: vccl.DataType.UINT8,
    torch.int32: vccl.DataType.INT32,
    torch.int64: vccl.DataType.INT64,
    torch.float16: vccl.DataType.FLOAT16,
    torch.float32: vccl.DataType.FLOAT32,
    torch.float64: vccl.DataType.FLOAT64,
    torch.bfloat16: vccl.DataType.BFLOAT16,
}


def _vccl_dtype(t: torch.Tensor) -> vccl.DataType:
    try:
        return _TORCH_DTYPE[t.dtype]
    except KeyError:
        raise TypeError(f"dtype {t.dtype} not supported by vccl") from None


class VcclCommunicator(DeviceCommunicatorBase):
    """DeviceCommunicator running collectives through vccl.

    Bootstrap: rank 0 of the group creates the vccl unique id and broadcasts
    it over the (gloo) cpu_group — the same pattern vLLM's pynccl uses.
    """

    def __init__(self, cpu_group: ProcessGroup, *args: object,
                 **kwargs: object) -> None:
        super().__init__(cpu_group, *args, **kwargs)

        uid_tensor = torch.empty(vccl.UNIQUE_ID_BYTES, dtype=torch.uint8)
        if self.rank_in_group == 0:
            uid = vccl.get_unique_id()
            uid_tensor.copy_(
                torch.frombuffer(bytearray(bytes(uid.internal)),
                                 dtype=torch.uint8))
        dist.broadcast(uid_tensor, src=self.ranks[0], group=self.cpu_group)
        uid = vccl.UniqueId.from_bytes(bytes(uid_tensor.numpy()))
        self._comm = vccl.Communicator(self.world_size, uid,
                                       self.rank_in_group)
        # (role, shape, dtype) -> (persistent tensor, vcclCommRegister handle).
        # M8: removes the per-call output allocation on the decode hot path.
        # H1(b): because the tensor is vcclCommRegister-ed, ScopedReg::add's
        # covers() check short-circuits on it, so a collective run on it does
        # zero ibv_reg_mr/ibv_dereg_mr (vs. ~10-50us/call on the volatile
        # activation tensor, whose address is not stable enough to register).
        # Decode shapes are few and stable, so this dict stays small; each
        # distinct (role, shape, dtype) seen holds one buffer for the life of
        # the communicator.
        self._buffers: dict = {}

    def _buf(self, role: str, shape, dtype: torch.dtype) -> torch.Tensor:
        key = (role, tuple(shape), dtype)
        got = self._buffers.get(key)
        if got is None:
            t = torch.empty(shape, dtype=dtype, device=self.device).contiguous()
            handle = self._comm.register(t)
            got = (t, handle)
            self._buffers[key] = got
        return got[0]

    # ── collectives ──────────────────────────────────────────────────────
    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        buf = self._buf("ar", input_.shape, input_.dtype)
        buf.copy_(input_)
        self._comm.all_reduce(buf, count=buf.numel(), dtype=_vccl_dtype(buf))
        input_.copy_(buf)
        return input_

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        if dim < 0:
            dim += input_.dim()
        input_size = input_.size()
        t = self._buf("ag_in", input_size, input_.dtype)
        t.copy_(input_)
        output = self._buf("ag_out", (self.world_size,) + tuple(input_size),
                           input_.dtype)
        self._comm.all_gather(t, output, sendcount=t.numel(),
                              dtype=_vccl_dtype(t))
        output = output.movedim(0, dim)
        return output.reshape(input_size[:dim]
                              + (self.world_size * input_size[dim],)
                              + input_size[dim + 1:])

    def reduce_scatter(self, input_: torch.Tensor,
                       dim: int = -1) -> torch.Tensor:
        # Mirrors DeviceCommunicatorBase.reduce_scatter's layout contract.
        world_size = self.world_size
        if world_size == 1:
            return input_
        assert -input_.dim() <= dim < input_.dim()
        if dim < 0:
            dim += input_.dim()
        moved = input_.movedim(0, dim)
        input_tensor = self._buf("rs_in", moved.shape, input_.dtype)
        input_tensor.copy_(moved)
        assert input_tensor.shape[0] % world_size == 0
        chunk_size = input_tensor.shape[0] // world_size
        output_shape = (chunk_size,) + input_tensor.shape[1:]
        output_tensor = self._buf("rs_out", output_shape, input_tensor.dtype)
        self._comm.reduce_scatter(input_tensor, output_tensor,
                                  recvcount=output_tensor.numel(),
                                  dtype=_vccl_dtype(input_tensor))
        return output_tensor.movedim(0, dim).contiguous()

    def broadcast(self, input_: torch.Tensor, src: int = 0) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        t = input_ if input_.is_contiguous() else input_.contiguous()
        self._comm.broadcast(t, root=src, count=t.numel(),
                             dtype=_vccl_dtype(t))
        if t is not input_:
            input_.copy_(t)
        return input_

    def send(self, tensor: torch.Tensor, dst: Optional[int] = None) -> None:
        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size
        t = tensor.contiguous()
        self._comm.send(t, dst, count=t.numel(), dtype=_vccl_dtype(t))

    def recv(self, size: torch.Size, dtype: torch.dtype,
             src: Optional[int] = None) -> torch.Tensor:
        if src is None:
            src = (self.rank_in_group - 1) % self.world_size
        tensor = torch.empty(size, dtype=dtype, device=self.device)
        self._comm.recv(tensor, src, count=tensor.numel(),
                        dtype=_vccl_dtype(tensor))
        return tensor

    def destroy(self) -> None:
        if getattr(self, "_comm", None) is not None:
            for _t, handle in self._buffers.values():
                self._comm.deregister(handle)
            self._buffers = {}
            self._comm.destroy()
            self._comm = None
