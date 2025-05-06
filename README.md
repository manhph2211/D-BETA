
<div align="center" style="font-size: 5em;">
  <strong>Boosting Masked ECG-Text Auto-Encoders as Discriminative Learners</strong>
  <br> </br> 
</div>

<div align="center"> 
<a href="https://maxph2211.github.io/D-BETA/"><img src="https://img.shields.io/badge/Website-CMELT WebPage-blue?style=for-the-badge"></a>
<a href="https://arxiv.org/pdf/2410.02131"><img src="https://img.shields.io/badge/arxiv-Paper-red?style=for-the-badge"></a>
<a href="https://huggingface.co/Manhph2211"><img src="https://img.shields.io/badge/Checkpoint-%F0%9F%A4%97%20Hugging%20Face-White?style=for-the-badge"></a>
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

## Introduction

This repo provides a quick example of running D-BETA with zero-shot setting on CODE-15 test dataset. It is structured as follows:

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

## Setups

First, we need to create a new Python environment and install the following:

```bash

* Python version 3.9

pip install -r requirements.txt
```

Next, please download the CODE-test data from [here](https://zenodo.org/records/3765780) and put it into in the `data/downstream/code-test` directory. 

Finally, we need to download the pre-trained model from [here](https://smu-my.sharepoint.com/:u:/g/personal/hm_pham_2023_phdcs_smu_edu_sg/EeuWpt1LaFFKkbWH5qmgSusB-XVE63LD9Xt66B2wdQyaaA?e=txcks6), and put it into `checkpoints` directory.

## Usage

To run the code, we can just use the `example.ipynb` notebook. 

```bibtex
@misc{pham2024cmeltcontrastiveenhancedmasked,
      title={Boosting Masked ECG-Text Auto-Encoders as Discriminative Learners}, 
      author={Manh Pham and Aaqib Saeed and Dong Ma},
      year={2024},
      eprint={2410.02131},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.02131}, 
}
```
