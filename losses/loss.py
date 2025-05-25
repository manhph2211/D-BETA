import math
import torch
import torch.nn.functional as F
from losses.ets import ETSLoss
from utils.metrics_utils import *


class DBETALoss:
    def __init__(self, cfg):
        super().__init__()
        self.norm_pix_loss = cfg.norm_pix_loss
        self.ets = ETSLoss()

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"]) 
        logits = model.get_logits(net_output)
        target = model.get_targets(sample, net_output, self.norm_pix_loss)
    
        losses = []
        reduction = "none" if not reduce else "mean"
        sample_size = 1
        
        loss = F.cross_entropy(
            logits["mlm_logits"],
            target["mlm_target"],
            ignore_index=-100,
            reduction=reduction,
        )
        losses.append(loss.detach().clone())

        mem_mask = net_output["mem_masks"]
        mem_loss = (logits["mem_logits"] - target["mem_target"]) ** 2
        mem_loss = mem_loss.mean(dim=-1)
        mem_loss = (mem_loss * mem_mask).sum() / mem_mask.sum()
        loss += mem_loss
        losses.append(mem_loss)
        
        etm_loss = F.cross_entropy(
            logits["etm_logits"],
            target["etm_target"],
            reduction=reduction,
        )
        loss += etm_loss
        losses.append(etm_loss)
        ets_loss = self.ets(logits["ets_uni_modal_feats"][0], logits["ets_uni_modal_feats"][1], target["etm_target"])
        loss += ets_loss
        losses.append(ets_loss)

        logging_output = {
            "loss": loss.item() if reduce else loss.detach(),
        }
        
        if len(losses) > 1:
            for i, l in enumerate(losses):
                logging_output[f"loss_{i}"] = l.item()
        
        with torch.no_grad():
            if logits["mlm_logits"].numel() == 0:
                mlm_corr = 0
                mlm_count = 0
            else:
                assert logits["mlm_logits"].dim() > 1, logits["mlm_logits"].shape
                mlm_corr = (logits["mlm_logits"].argmax(-1) == target["mlm_target"]).long().sum().item()
                mlm_count = (target['mlm_target'] != -100).sum()
                
                assert mlm_corr <= mlm_count
            logging_output["mlm_correct"] = mlm_corr
            logging_output["mlm_count"] = mlm_count
            
            if logits["etm_logits"].numel() == 0:
                etm_corr = 0
                etm_count = 0
            else:
                assert logits["etm_logits"].dim() > 1, logits["etm_logits"].shape
                etm_corr = (logits["etm_logits"].argmax(-1) == target["etm_target"]).long().sum().item()
                etm_count = target["etm_target"].numel()
            logging_output["etm_correct"] = etm_corr
            logging_output["etm_count"] = etm_count
        
        return loss, sample_size, logging_output
