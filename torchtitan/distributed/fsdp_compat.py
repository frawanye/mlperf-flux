# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FSDP2 compatibility layer for different PyTorch builds.

Some PyTorch builds (particularly ROCm builds) have FSDP2 under
torch.distributed._composable.fsdp instead of torch.distributed.fsdp.
This module provides a unified import path.
"""

try:
    # Try the standard FSDP2 path first (PyTorch 2.4+ stable)
    from torch.distributed.fsdp import (
        CPUOffloadPolicy,
        fully_shard,
        MixedPrecisionPolicy,
    )
except ImportError:
    # Fall back to composable path (ROCm builds, some nightlies)
    from torch.distributed._composable.fsdp import (
        CPUOffloadPolicy,
        fully_shard,
        MixedPrecisionPolicy,
    )

# Also export FSDPModule if available
try:
    from torch.distributed.fsdp import FSDPModule
except ImportError:
    try:
        from torch.distributed._composable.fsdp import FSDPModule
    except ImportError:
        # FSDPModule might not exist in all versions
        FSDPModule = None

__all__ = ["CPUOffloadPolicy", "fully_shard", "MixedPrecisionPolicy", "FSDPModule"]

