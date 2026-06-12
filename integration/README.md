# vllm-vulkan integration

`vllm-vulkan-vccl.patch` wires vccl into
[vllm-vulkan](https://github.com/ericcurtin/vllm-vulkan) as the device
communicator, enabling multi-node tensor parallelism (one Vulkan device per
node) over RoCE v2 RDMA with automatic TCP fallback.

What the patch changes in `vllm_vulkan/platform.py`:

- `VulkanPlatform.get_device_communicator_cls()` returns
  `vccl.torch_comm.VcclCommunicator` when the `vccl` Python package is
  importable, falling back to vLLM's base (gloo) communicator otherwise.
- `dist_backend = "gloo"`: torch.distributed still bootstraps the process
  groups; only the data-plane collectives go through vccl.
- `distributed_executor_backend` is no longer forced to `"uni"` when
  `world_size > 1` (defaults to `"mp"`; multi-node launches typically pass
  `--distributed-executor-backend external_launcher` or use ray).

## Apply

```sh
git clone https://github.com/ericcurtin/vllm-vulkan
cd vllm-vulkan
git apply /path/to/vccl/integration/vllm-vulkan-vccl.patch
```

Install the vccl pieces on every node:

```sh
cmake -B build && cmake --build build -j && sudo cmake --install build
pip install /path/to/vccl/python
```

## Verify

`python/tests/test_e2e_tp.py` runs a tensor-parallel forward pass
(column/row-parallel MLP with all-reduce, column-parallel LM head with
all-gather, pipeline send/recv) through `VcclCommunicator` and compares
against the unsharded reference. With `VLLM_VULKAN_DIR` pointing at the
patched checkout it also asserts the platform resolves to vccl:

```sh
VLLM_VULKAN_DIR=/path/to/vllm-vulkan python3 python/tests/test_e2e_tp.py
```

## Multi-node notes

- Set `VCCL_SOCKET_IFNAME`/`VCCL_SOCKET_ADDR` to the interface the nodes
  reach each other on; the verbs transport picks the RoCE v2 IPv4-mapped GID
  automatically (`VCCL_GID_INDEX` to override).
- `VCCL_TRANSPORT=tcp` forces the fallback transport for debugging.
- vccl operates on host-visible memory. vllm-vulkan keeps PyTorch tensors on
  CPU, so this covers all collectives today; on unified-memory APUs
  (BC-250) it already includes GPU-visible data.
