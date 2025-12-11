# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass, field

import torchtitan.experiments.flux.job_config as jc


@dataclass
class Preprocessing:
    # Path should be set via --preprocessing.output_dataset_path CLI arg
    output_dataset_path: str = ""


@dataclass
class JobConfig(jc.JobConfig):
    preprocessing: Preprocessing = field(default_factory=Preprocessing)
