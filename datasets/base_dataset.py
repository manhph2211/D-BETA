# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import numpy as np
import torch.utils.data

logger = logging.getLogger(__name__)


class EpochListening:
    """Mixin for receiving updates whenever the epoch increments."""

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        """
        Whether we can reuse the :class:`fairseq.data.EpochBatchIterator` for
        this dataset across epochs.

        This needs to return ``False`` if the sample size can change across
        epochs, in which case we may need to regenerate batches at each epoch.
        If your dataset relies in ``set_epoch`` then you should consider setting
        this to ``False``.
        """
        return True
    
    def set_epoch(self, epoch):
        """Will receive the updated epoch number at the beginning of the epoch."""
        pass

class BaseDataset(torch.utils.data.Dataset, EpochListening):
    """A dataset that provides helpers for batching."""

    def __getitem__(self, index):
        raise NotImplementedError
    
    def __len__(self):
        raise NotImplementedError
    
    def collator(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate
        
        Returns:
            dict: a mini-batch suitable for forwarding with a Model
        """
        raise NotImplementedError
    
    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        raise NotImplementedError

    def num_tokens_vec(self, indices):
        """Return the number of tokens for a set of positions defined by indices.
        This value is used to enforce ``--max-tokens`` during batching."""
        raise NotImplementedError

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        raise NotImplementedError

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        return np.arange(len(self), dtype = np.int64)
    
    @property
    def supports_prefetch(self):
        """Whether this dataset supports prefetching."""
        return False
    
    def attr(self, attr: str, index: int):
        return getattr(self, attr, None)
    
    def prefetch(self, indices):
        """Prefetch the data required for this epoch."""
        return NotImplementedError

    def filter_indices_by_size(self, indices, max_sizes):
        """
        Filter a list of sample indices. Remove those that are longer than
        specified in *max_sizes*.

        WARNING: don't update, override method in child classes

        Args:
            indices (np.array): original array of sample indices
            max_sizes (int or list[int] or tuple[int]): max sample size,
                can be defined separately for src and tgt (then list or tuple)

        Returns:
            np.array: filtered sample array
            list: list of removed indices
        """
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
        """Whether this dataset supports fetching outside the works of the dataloader."""
        return True