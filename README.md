
<div align="center" style="font-size: 5em;">
  <strong>Boosting Masked ECG-Text Auto-Encoders as Discriminative Learners (ICML 2025)</strong>
  <br> </br> 
</div>

<div align="center"> 
<a href="https://manhph2211.github.io/D-BETA/"><img src="https://img.shields.io/badge/Website-DBETA WebPage-blue?style=for-the-badge"></a>
<a href="https://arxiv.org/pdf/2410.02131"><img src="https://img.shields.io/badge/arxiv-Paper-red?style=for-the-badge"></a>
<a href="https://huggingface.co/Manhph2211/D-BETA"><img src="https://img.shields.io/badge/Checkpoint-%F0%9F%A4%97%20Hugging%20Face-White?style=for-the-badge"></a>
</div>

<div align="center">
  <a href="https://maxph2211.dev/" target="_blank">Hung&nbsp;Manh&nbsp;Pham</a> &emsp;
  <a href="https://aqibsaeed.github.io/" target="_blank">Aaqib&nbsp;Saeed</a> &emsp;
  <a href="https://www.dongma.info/" target="_blank">Dong&nbsp;Ma</a> &emsp;
</div>
<br>

<div align="center">
    <img src="assets/D-BETA.gif" alt="Illustration of our contrastive masked ECG-language modeling technique"/>
</div>

## :rocket: Introduction

**What about an ECG foundation model?**

Cardiovascular diseases are the leading cause of death worldwide, accounting for an estimated 17.9 million deaths annually, which is about 32% of all global deaths. Electrocardiograms (ECGs) play a crucial role in diagnosing these conditions, with over 300 million ECGs performed each year globally.

Despite the widespread use of ECGs, there's a lack of publicly available general-purpose models that can effectively interpret ECG data across diverse populations and conditions. Our work presents D-BETA, an approach that learns general knowledge from both ECG signals and their relevant textual reports simultaneously without needing exact manual labels during pre-training. D-BETA not only captures subtle details in each type of data but also learns how they connect, helping it make a better foundation model with more accurate decisions in downstream tasks.

Across comprehensive evaluation, D-BETA consistently outperforms strong baselines on multiple cardiac conditions, offering a scalable, self-supervised path toward accurate, label-efficient heart health AI worldwide.

This repository shows how to perform inference with the model and includes a quick example in a zero-shot setting on the CODE-15 test dataset. The structure is as follows:

```angular2html
.
├── configs
│   ├── config.json
├── data
│   ├── pretrain
│   ├── downstream
│   │   ├── code-test
│   │   │   └── data
│   │           ├── annotations
│   │           ├── ecg_tracings.hdf5
├── models
│   ├── modules
│   └── dbeta.py
└── example.ipynb
└── README.md

```

## :book: Usage

### Load with `transformers 4.36.2`

```python
from transformers import AutoModel
import torch

model = AutoModel.from_pretrained("Manhph2211/D-BETA", trust_remote_code=True)
model.eval()

ecgs = torch.randn(2, 12, 5000) # [batch, leads, length]
with torch.no_grad():
    output = model(ecgs)

ecg_features = output.pooler_output
print(ecg_features.shape)  # (2, 768)
```

### Load with the GitHub repo

Clone the project and prepare the environment:

```bash
git clone https://github.com/manhph2211/D-BETA.git && cd D-BETA
conda create -n dbeta python=3.9
conda activate dbeta
pip install -r requirements.txt
```

Checkout our `example.ipynb` notebook for a quick example of using the model for zero-shot classification on the CODE-15 test dataset. You can also use the encoder directly:

```python
import torch
from models.processor import get_model, get_ecg_feats

model = get_model(config_path='configs/config.json', checkpoint_path='checkpoints/pytorch_model.bin') # or sample.pt
ecgs = torch.randn(2, 12, 5000)  # [batch, leads, length]
ecg_features = get_ecg_feats(model, ecgs)
print(ecg_features.shape)  # (2, 768)
```


## :memo: Acknowledgments

This research was supported by the Google South Asia & Southeast Asia research award.

We are also thankful for the valuable work provided by this nice [repo](https://github.com/Jwoo5/fairseq-signals) and [repo](https://github.com/cheliu-computation/MERL-ICML2024).

## :page_facing_up: Citation

If you find this work useful :smile:, please consider citing our paper:

```bibtex
@inproceedings{
  hung2025boosting,
  title={Boosting Masked {ECG}-Text Auto-Encoders as Discriminative Learners},
  author={Manh Pham Hung and Aaqib Saeed and Dong Ma},
  booktitle={Forty-second International Conference on Machine Learning},
  year={2025},
}
```
