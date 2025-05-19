import os
import sys
import logging

import scipy.io
import numpy as np
import torch
from transformers import (
    DataCollatorForWholeWordMask,
    DataCollatorForLanguageModeling,
    BertTokenizerFast,
    DebertaTokenizerFast,
    T5TokenizerFast,
    RobertaTokenizerFast
)
import faiss
import json
from transformers import AutoTokenizer, AutoModel, T5EncoderModel
from datasets.run_faiss import faiss_read

from datasets.ecg_dataset import RawECGDataset
import warnings
warnings.filterwarnings("ignore")
import urllib3
urllib3.disable_warnings()

logger = logging.getLogger(__name__)


class RawECGTextDataset(RawECGDataset):
    def __init__(
        self,
        max_text_size=256,
        tokenizer="", 
        compute_mlm_indices=False,
        mlm_prob=0.15,
        medvill=False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.tokenizer = T5TokenizerFast.from_pretrained(
            tokenizer, do_lower_case="uncased" in tokenizer
        ) 
        
        self.faiss_tokenizer = T5TokenizerFast.from_pretrained("google/flan-t5-small") # NOTE
        self.faiss_lm = T5EncoderModel.from_pretrained("google/flan-t5-small") # NOTE
        self.faiss_index = faiss.read_index(f"data/mimic_iv_ecg_index.faiss")

        with open(f'data/mimic_iv_ecg_index.json', 'r') as f:
            self.faiss_embedding_to_sample_map = json.load(f)
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_token = self.tokenizer.pad_token_id
        self.sep_token = self.tokenizer.sep_token_id

        self.max_text_size = (
            max_text_size - 2 if max_text_size is not None else sys.maxsize
        )
        self.min_text_size = 0
        if medvill:
            self.min_text_size = 10

        self.compute_mlm_indices = compute_mlm_indices
        self.mlm_prob = mlm_prob
        if self.compute_mlm_indices:
            self.mlm_collator = DataCollatorForLanguageModeling( 
                tokenizer=self.tokenizer,
                mlm=False, 
                mlm_probability=self.mlm_prob
            )
        
        self.medvill = medvill
        
    def normalize_text(self, text):
        import re # let's preprocess a bit :)
        if isinstance(text, str):
            text = text.lower()
            text = text.strip()
            report = text.replace('ekg', 'ecg')
            report = report.strip('*** ')
            report = report.strip(' ***')
            report = report.strip('***')
            report = report.strip('=-')
            report = report.strip('=')
        return text

    def collator(self, samples):
        texts = []    
        for s in samples:
            texts.append(self.normalize_text(s["text"]))

        if len(texts) == 0:
            return {}

        encodings = self.tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=self.max_text_size,
            return_special_tokens_mask=True,
            return_offsets_mapping=True,
        )
        
        max_len = len(encodings["input_ids"][0])

        collated_texts = torch.LongTensor(encodings["input_ids"])
        text_padding_mask = ~torch.BoolTensor(encodings["attention_mask"])

        text_sizes = [(x != 0).sum().item() for x in collated_texts]
        valid_indices = [i for i, x in enumerate(text_sizes) if x > self.min_text_size]

        samples = [x for i, x in enumerate(samples) if i in valid_indices]
        collated_texts = collated_texts[valid_indices]
        text_padding_mask = text_padding_mask[valid_indices]

        _collated_ecgs = super().collator(
            [{'source': s['ecg'], 'id': s['id'], 'original': s['original']} for s in samples]
        )
        if len(_collated_ecgs) == 0:
            return {}

        collated_ecgs = _collated_ecgs['net_input']['source']
        ecg_padding_mask = _collated_ecgs['net_input']['padding_mask']

        input = {
            'ecg': collated_ecgs,
            'ecg_padding_mask': ecg_padding_mask,
        }

        if "text" in samples[0]: 
            bsz = len(samples)
            is_aligned = torch.ones((bsz,))
            
            num_negatives = int((bsz * 0.5 + np.random.rand()))

            neg_idcs = np.random.choice(bsz, size=num_negatives, replace=False)
            
            for i in neg_idcs:
                text = self.normalize_text(samples[i]["text"])
                least_sim_texts = faiss_read(text, self.faiss_lm, self.faiss_tokenizer, self.faiss_index, self.faiss_embedding_to_sample_map, 64)
                random_negative = least_sim_texts 
                ids = self.tokenizer(
                    random_negative,
                    padding="max_length",
                    truncation=True,
                    max_length=max_len,
                    return_special_tokens_mask=True,
                    return_offsets_mapping=True,
                )
                collated_texts[i] = torch.LongTensor(ids["input_ids"])
                text_padding_mask[i] = ~torch.BoolTensor(ids["attention_mask"])
                is_aligned[i] = 0

        if self.compute_mlm_indices:
            flatten_encodings = [
                {key: encodings[key][i] for key in encodings.keys()}
                for i in range(len(encodings["input_ids"]))
            ]
            collated_encodings = self.mlm_collator(flatten_encodings)
            collated_texts = collated_encodings["input_ids"]
            
            input["text"] = collated_texts
        else:
            input["text"] = collated_texts

        text_attention_mask = ~text_padding_mask
        input["text_padding_mask"] = text_padding_mask
        input["text_attention_mask"] = text_attention_mask

        out = {
            'id': torch.LongTensor([s['id'] for s in samples]),
            'net_input': input
        }
        if "text" in samples[0]: 
            out["is_aligned"] = is_aligned
        if self.compute_mlm_indices:
            out["mlm_labels"] = collated_encodings["labels"]

        return out

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""

        if self.shuffle:
            order = [np.random.permutation(len(self))]
            return order[0]
        else:
            return np.arange(len(self))
    
    def sample_negative(self, dx, candidates, diagnoses):
        for cand in candidates:
            if dx != diagnoses[cand]:
                return cand
        return None


class ProcessedECGTextDataset(RawECGTextDataset):
    def __init__(self, manifest_path, num_buckets=0, **kwargs):
        super().__init__(**kwargs)

        skipped = 0
        self.fnames = []
        sizes = []
        self.skipped_indices = set()

        with open(manifest_path, 'r') as f:
            first_line = f.readline().strip() 
            self.root_dir = "data/pretrain/processed_data"
            for i, line in enumerate(f):
                items = line.strip().split('\t')
                assert len(items) == 2, line
                ecg_sz = 5000
                if self.min_sample_size is not None and ecg_sz < self.min_sample_size:
                    skipped += 1
                    self.skipped_indices.add(i)
                    continue
                self.fnames.append(items[0])
                sizes.append(ecg_sz)
        
        logger.info(f"loaded {len(self.fnames)}, skipped {skipped} samples")

        self.sizes = np.array(sizes, dtype=np.int64)

        try:
            import pyarrow

            self.fnames = pyarrow.array(self.fnames)
        except:
            logger.debug(
                "Could not create a pyarraw array. Please install pyarrow for better performance"
            )
            pass

        self.set_bucket_info(num_buckets)

    def __getitem__(self, index):
        path = os.path.join(self.root_dir, str(self.fnames[index]))

        res = {'id': index}

        data = scipy.io.loadmat(path)

        curr_sample_rate = data['curr_sample_rate']
        feats = torch.from_numpy(data['feats'])
        res["original"] = feats
        res["ecg"] = self.postprocess(feats, curr_sample_rate)
        res["text"] = self.normalize_text(data["text"][0])
        if "diagnoses" in data: 
            res["diagnoses"] = [x.strip() for x in data["text"]]

        return res

    def __len__(self):
        return len(self.fnames)