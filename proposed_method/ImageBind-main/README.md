# Towards Open-Vocabulary Audio-Visual Event Localization

Jinxing Zhou, Dan Guo, Ruohao Guo, Yuxin Mao, Jingjing Hu, Yiran Zhong, Xiaojun Chang, Meng Wang

Official code for our paper: [Towards Open-Vocabulary Audio-Visual Event Localization](https://arxiv.org/abs/2411.11278)



## Data Preparation

The proposed OV-AVEBench dataset is available for the community now. You may directly download the preprocessed audio (.wav) and visual (.png) files from [this link](https://mailhfuteducn-my.sharepoint.com/:u:/g/personal/2018110964_mail_hfut_edu_cn/Ef9AH0VrrVFGlocbYQUiFpEBa-afOfGFDuctUhCQqVKFDw?e=PiQwOT) to develop your own models for OV-AVEL task.  The raw videos are also available at [here](https://mailhfuteducn-my.sharepoint.com/:u:/g/personal/2018110964_mail_hfut_edu_cn/EcVHOp2zOyVHvi1Au-i1zFQBf5wQNi-Yff9Aso_SJ4MV8Q?e=OeRlQh).

Please put the downloaded preprocessed data into `ovave_dataset_preprocessed' directory.



## Training-free Baseline


```script
bash run_baseline_v0.sh
```


## Fine-tuning Baseline

```script
bash run_baseline_v1_train_fully.sh
```


## Citation
If our work is helpful for your research, please consider citing it:
```script
@article{zhou2024towards,
  title={Towards Open-Vocabulary Audio-Visual Event Localization},
  author={Zhou, Jinxing and Guo, Dan and Guo, Ruohao and Mao, Yuxin and Hu, Jingjing and Zhong, Yiran and Chang, Xiaojun and Wang, Meng},
  journal={arXiv preprint arXiv:2411.11278},
  year={2024}
}
```