import torch
import os
import shutil
from yaml import safe_dump
import time
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import random
from utils.utils import save_img,setup_seed, test_fix_resolution, test_unfix_resolution
from torchinfo import summary
from torch.utils.data import random_split
from utils.yml import NoiseOptions, parse_yml, dict_to_nonedict
from PIL import Image
from pathlib import Path
import json
import argparse
from datetime import datetime

from utils.dataloader import build_dataset
from utils.dataloader import build_dataset,build_dataset_MSCOCO

#some imports you need to custimize with your own model, for loading your model
from utils.train_param import TrainParam
from models import uvit,convnext,jnd


if __name__=="__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    
    #Please modify the factors of each noise in this yaml
    noise_opt = NoiseOptions("./noise_opt.yml")
    noise_opt = noise_opt.get_noise_opt()
    option_yml = parse_yml("noise_opt.yml")
    opt = dict_to_nonedict(option_yml)

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--datapath', type=str)
    parser.add_argument('--fix_res_test',  default=False, action='store_true')
    parser.add_argument('--input_res', default=256, type=int)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--message', type=int, default=32)
    args = parser.parse_args()
    setup_seed(args.seed)

    #load your model
    file_path  = Path(args.ckpt)
    dir_path = file_path.parent
    with open(os.path.join(dir_path,"train_param.json"), 'r') as f:
        param_dict = json.load(f) 
    
    params = TrainParam(**param_dict)
    params.batchsize = 8
    encoder = uvit.Unet1(params)
    decoder = convnext.ConvNextDecoder(params)
    ckpt = torch.load(args.ckpt,map_location=torch.device('cpu'))
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    encoder.eval()
    decoder.eval()
    encoder.to(device)
    decoder.to(device)
    del ckpt
    image_mean = torch.tensor([0.5, 0.5, 0.5])
    image_std = torch.tensor([0.5, 0.5, 0.5])

    jnd_ = jnd.JND(preprocess=transforms.Normalize(-image_mean / image_std, 1 / image_std), 
                          postprocess=transforms.Normalize(image_mean, image_std)).to(device)
    
    encoder.scaling_i = params.scaling_i
    encoder.scaling_w = params.scaling_w
    #data prepare
    if not  args.fix_res_test:
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
        dataset = build_dataset(args.datapath,trans=trans)
        dataloder = DataLoader(dataset,batch_size=1,shuffle=False,num_workers=4,drop_last=True) #different resolution could not concatenate
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = f"./Validation_Results/Unfix{timestamp}"
        os.makedirs(save_path, exist_ok=True)
        shutil.copy("noise_opt.yml", save_path)

        # test method
        test_unfix_resolution(encoder,decoder,noise_opt,dataloder, 
                              save_path,device,resolution=args.input_res,message=args.message, jnd=jnd_)
    else:
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(args.input_res),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])  
        dataset = build_dataset(args.datapath,trans=trans)
        dataloder = DataLoader(dataset,batch_size=params.batchsize,shuffle=False,num_workers=4,drop_last=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = f"./Validation_Results/Fix{timestamp}"
        os.makedirs(save_path, exist_ok=True)
        shutil.copy("noise_opt.yml", save_path)

        test_fix_resolution(encoder,decoder,noise_opt,dataloder,
                              save_path,device,resolution=args.input_res,message=args.message,jnd=jnd_)