# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import numpy as np
import torch.utils.data

logger = logging.getLogger(__name__)


class EpochListening:
    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return True
    
    def set_epoch(self, epoch):
        pass

class BaseDataset(torch.utils.data.Dataset, EpochListening):
    def __getitem__(self, index):
        raise NotImplementedError
    
    def __len__(self):
        raise NotImplementedError
    
    def collator(self, samples):
        raise NotImplementedError
    
    def num_tokens(self, index):
        raise NotImplementedError

    def num_tokens_vec(self, indices):
        raise NotImplementedError

    def size(self, index):
        raise NotImplementedError

    def ordered_indices(self):
        return np.arange(len(self), dtype = np.int64)
    
    @property
    def supports_prefetch(self):
        return False
    
    def attr(self, attr: str, index: int):
        return getattr(self, attr, None)
    
    def prefetch(self, indices):
        return NotImplementedError

    def filter_indices_by_size(self, indices, max_sizes):
        if isinstance(max_sizes, float) or isinstance(max_sizes, int):
            if hasattr(self, "sizes") and isinstance(self.sizes, np.ndarray):
                ignored = indices[self.sizes[indices] > max_sizes].tolist()
                indices = indices[self.sizes[indices] <= max_sizes]
            elif (
                hasattr(self, "sizes")
                and isinstance(self.sizes, list)
                and len(self.sizes) == 1
            ):
                ignored = indices[self.sizes[0][indices] > max_sizes].tolist()
                indices = indices[self.sizes[0][indices] <= max_sizes]
            else:
                pass
        else:
            pass
        return indices, ignored

    @property
    def support_fetch_outside_dataloader(self):
        return True