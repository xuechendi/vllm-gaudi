from vllm.distributed.kv_transfer.kv_connector.v1 import nixl_connector
import torch
from vllm.logger import init_logger
logger = init_logger(__name__)

nixl_connector._NIXL_SUPPORTED_XPUS = {
    "cuda": ("cuda", ),
    "hpu": ("cpu", ),
    "tpu": ("cpu", ),
}

def initialize_host_xfer_buffer(
        self, kv_caches: dict[str, torch.Tensor]) -> None:
    """
    Initialize transfer buffer in CPU mem for accelerators
    NOT directly supported by NIXL (e.g., tpu)
    """
    xfer_buffers: dict[str, torch.Tensor] = {}
    try:
        for layer_name, kv_cache in kv_caches.items():
            if isinstance(kv_cache, tuple):
                kv_shape = (2, *kv_cache[0].shape)
                kv_dtype = kv_cache[0].dtype
                xfer_buffers[layer_name] = torch.empty(kv_shape,
                                                    dtype=kv_dtype,
                                                    device="cpu")
            else:
                kv_shape = kv_cache.shape
                kv_dtype = kv_cache.dtype
                xfer_buffers[layer_name] = torch.empty(kv_shape,
                                                    dtype=kv_dtype,
                                                    device="cpu")
    except MemoryError as e:
        logger.error("NIXLConnectorWorker gets %s.", e)
        raise

    self.host_xfer_buffers = xfer_buffers

nixl_connector.NixlConnectorWorker.initialize_host_xfer_buffer = initialize_host_xfer_buffer