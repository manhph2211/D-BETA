import torch
import torch.nn.functional as F
from losses.ets import ETSLoss


class DBETALoss:
    def __init__(self, cfg):
        self.norm_pix_loss = cfg.norm_pix_loss
        self.ets = ETSLoss()

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])
        logits = model.get_logits(net_output)
        target = model.get_targets(sample, net_output, self.norm_pix_loss)

        reduction = "mean" if reduce else "none"
        total_loss = 0
        losses = []

        # MLM loss
        mlm_logits, mlm_target = logits["mlm_logits"], target["mlm_target"]
        mlm_loss = F.cross_entropy(mlm_logits, mlm_target, ignore_index=-100, reduction=reduction)
        total_loss += mlm_loss
        losses.append(mlm_loss)

        # MEM loss
        mem_logits, mem_target, mem_mask = logits["mem_logits"], target["mem_target"], net_output["mem_masks"]
        mem_loss = ((mem_logits - mem_target) ** 2).mean(dim=-1)
        mem_loss = (mem_loss * mem_mask).sum() / mem_mask.sum()
        total_loss += mem_loss
        losses.append(mem_loss)

        # ETM loss
        etm_logits, etm_target = logits["etm_logits"], target["etm_target"]
        etm_loss = F.cross_entropy(etm_logits, etm_target, reduction=reduction)
        total_loss += etm_loss
        losses.append(etm_loss)

        # ETS loss
        feat1, feat2 = logits["ets_uni_modal_feats"]
        ets_loss = self.ets(feat1, feat2, etm_target)
        total_loss += ets_loss
        losses.append(ets_loss)

        logging_output = {"loss": total_loss.item() if reduce else total_loss.detach()}
        for i, l in enumerate(losses):
            logging_output[f"loss_{i}"] = l.item()

        with torch.no_grad():
            if mlm_logits.numel() == 0:
                mlm_corr = mlm_count = 0
            else:
                mlm_pred = mlm_logits.argmax(-1)
                mlm_mask = mlm_target != -100
                mlm_corr = (mlm_pred == mlm_target)[mlm_mask].sum().item()
                mlm_count = mlm_mask.sum().item()
            logging_output.update({"mlm_correct": mlm_corr, "mlm_count": mlm_count})

            if etm_logits.numel() == 0:
                etm_corr = etm_count = 0
            else:
                etm_corr = (etm_logits.argmax(-1) == etm_target).sum().item()
                etm_count = etm_target.numel()
            logging_output.update({"etm_correct": etm_corr, "etm_count": etm_count})

        return total_loss, logging_output
