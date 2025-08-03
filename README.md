
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
  <a href="https://manhph2211.github.io/" target="_blank">Hung&nbsp;Manh&nbsp;Pham</a> &emsp;
  <a href="https://aqibsaeed.github.io/" target="_blank">Aaqib&nbsp;Saeed</a> &emsp;
  <a href="https://www.dongma.info/" target="_blank">Dong&nbsp;Ma</a> &emsp;
</div>
<br>

<div align="center">
    <img src="assets/D-BETA.gif" alt="Illustration of our contrastive masked ECG-language modeling technique"/>
</div>

## :rocket: Introduction

**What about an ECG signal foundation model?**

Cardiovascular diseases are the leading cause of death worldwide, accounting for an estimated 17.9 million deaths annually, which is about 32% of all global deaths. Electrocardiograms (ECGs) play a crucial role in diagnosing these conditions, with over 300 million ECGs performed each year globally.

Despite the widespread use of ECGs, there's a lack of general-purpose models that can effectively interpret ECG data across diverse populations and conditions. Our work presents D-BETA, a new approach that learns directly from both ECG signals and their relevant textual reports simultaneously without needing exact manual labels. D-BETA not only captures subtle details in each type of data but also learns how they connect, helping it make a better foundation model with more accurate decisions.

Across comprehensive evaluation, D-BETA consistently outperforms strong baselines on 100+ cardiac conditions, offering a scalable, self-supervised path toward accurate, label-efficient heart health AI worldwide.

This repo provides a quick example of running D-BETA with a zero-shot experiment on CODE-15 test dataset. It is structured as follows:

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
└── infer.ipynb
└── README.md

```

## :book: Usage

First, we need to clone the project and prepare the environment as follows:

```bash
git clone https://github.com/manhph2211/D-BETA.git && cd D-BETA
conda create -n dbeta python=3.9
conda activate dbeta
pip install -r requirements.txt
```

Next, please download the CODE-test data from [here](https://zenodo.org/records/3765780) and put it into the `data/downstream/code-test` directory. 

Then, we need to download the pre-trained model from [here](https://huggingface.co/Manhph2211/D-BETA), and put it into `checkpoints` directory.

Finally, to run the code, we can just use the `example.ipynb` notebook. You can also run the following command to execute the encoder only for feature extraction:

```python

import torch
from models.processor import get_model, get_ecg_feats

model = get_model(config_path='configs/config.json', checkpoint_path='checkpoints/sample.pt')
ecgs = torch.randn(2, 12, 5000)  # [batch, leads, length], 5000 = 10s x 500Hz 
ecg_features = get_ecg_feats(model, ecgs)
print(ecg_features.shape) # (2, 768)

```

## :memo: Acknowledgments

This research was supported by the Google South Asia & Southeast Asia research award.

We are also thankful for the valuable work provided by this nice [repo](https://github.com/Jwoo5/fairseq-signals) and [repo](https://github.com/cheliu-computation/MERL-ICML2024).

## :page_facing_up: Citation

If you find this work useful :smile:, please consider citing our paper:

```bibtex
@inproceedings{hungboosting,
  title={Boosting Masked ECG-Text Auto-Encoders as Discriminative Learners},
  author={Hung, Manh Pham and Saeed, Aaqib and Ma, Dong},
  booktitle={Forty-second International Conference on Machine Learning}
}
```
