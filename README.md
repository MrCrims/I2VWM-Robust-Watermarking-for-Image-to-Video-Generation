# [arxiv] I2VWM: Robust Watermarking for Image to Video Generation
Official implementation of [I2VWM: Robust Watermarking for Image to Video Generation](https://arxiv.org/abs/2509.17773).



1. Download the data ([dataset](https://drive.google.com/drive/folders/1QnSVKztpwiSrsmY6efgUctOm2_MLoKjw?usp=sharing), [prompts](https://drive.google.com/file/d/1Sv7Yk30Bqxh-XcdSVhsHq_lPyalAS53F/view?usp=sharing)) and put them into the data dir `./data`. Download the checkpoinits([param](https://drive.google.com/file/d/18zOaxC1SFJVoJQjBcgr1Q2ZMYx53MW_P/view?usp=sharing), [checkpoinits](https://drive.google.com/file/d/1rlm4BNoKEMn4dl8oUFJ6n5f0CIS3yxVq/view?usp=sharing)) and put them into the data dir `./checkpoinits`.

2. Test classic noise

```
python test_tradition_noise.py --ckpt ./checkponits --datapath data/DIV2K_valid_HR --message 32 
```

3. Test I2V. 


```
python test_I2V.py --ckpt ./checkpoints  --datapath data/DIV2K_valid_HR --message 32 --mode encode --seed 42

python test_I2V.py --datapath "resluts of encode image usually in "Validation Results"" --mode I2V --test_num 10 --seed 58

python test_I2V.py --datapath "resluts of encode image usually in "Validation Results"" --message 32 --mode decode
```


## Acknowledgements
This code builds on the code from the [diffusers](https://github.com/huggingface/diffusers) library, [Watermark-Anything Model](https://github.com/facebookresearch/watermark-anything), and [TrustMark](https://github.com/adobe/trustmark). 

## Cite
If you find this repository useful, please consider giving a star ⭐ and please cite as:
```
@misc{wang2025i2vwmrobustwatermarkingimage,
      title={I2VWM: Robust Watermarking for Image to Video Generation}, 
      author={Guanjie Wang and Zehua Ma and Han Fang and Weiming Zhang},
      year={2025},
      eprint={2509.17773},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.17773}, 
}

```







