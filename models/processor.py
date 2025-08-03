from copy import deepcopy
from omegaconf import OmegaConf
import torch
from tqdm import tqdm
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import json
import sys
sys.path.append(".")
from models.dbeta import DBETA
from types import SimpleNamespace


def get_model(config_path, checkpoint_path) :
    with open(config_path, 'r') as json_file:
        cfg = json.load(json_file)

    cfg = SimpleNamespace(**cfg['model'])
    model = DBETA(cfg)
    try:
        checkpoint = torch.load(checkpoint_path)
        if "ecg_encoder.mask_emb" in checkpoint["model"].keys():
            del checkpoint["model"]["ecg_encoder.mask_emb"]

        model.load_state_dict(checkpoint["model"], strict=True)
        print("Loaded pre-trained ECG encoder ...")
    except:
        print("Not using pre-trained checkpoints!!!")
        
    model.remove_pretraining_modules()
    return model


def get_ecg_feats(model, ecgs, args=None):        

    uni_modal_ecg_feats, ecg_padding_mask = (
        model.ecg_encoder.get_embeddings(ecgs, padding_mask=None)
    )
    
    cls_emb = model.class_embedding.repeat((len(uni_modal_ecg_feats), 1, 1))
    uni_modal_ecg_feats = torch.cat([cls_emb, uni_modal_ecg_feats], dim=1)
    uni_modal_ecg_feats = model.ecg_encoder.get_output(uni_modal_ecg_feats, ecg_padding_mask)
    out = model.multi_modal_ecg_proj(uni_modal_ecg_feats)
    ecg_features = model.unimodal_ecg_pooler(out)
    return ecg_features


if __name__ == "__main__":
    model = get_model(config_path='configs/config.json', checkpoint_path='checkpoints/sample.pt')
    ecgs = torch.randn(2, 12, 5000)  # [batch, leads, length], 5000 = 10s x 500Hz 
    ecg_features = get_ecg_feats(model, ecgs)
    print(ecg_features.shape) # (2, 768)
    
