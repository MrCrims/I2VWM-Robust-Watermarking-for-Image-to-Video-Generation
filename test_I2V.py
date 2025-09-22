import torch
import os
import shutil
from yaml import safe_dump
import time
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import random
from utils.utils import *
from torchinfo import summary
from torch.utils.data import random_split
from utils.yml import NoiseOptions, parse_yml, dict_to_nonedict
from PIL import Image
from pathlib import Path
import json
import argparse
import logging
from datetime import datetime
from utils.video_generation import choose_resolution, choose_test, load_video, load_image, export_to_video
from typing import Callable, List, Literal, Optional

from utils.dataloader import build_dataset_I2V

#some imports you need to custimize with your own model, for loading your model
from utils.train_param import TrainParam
from models import uvit,convnext,jnd

@torch.no_grad
def encode(datapath, save_path, device, **args):
    #load your model
    file_path  = Path(args['ckpt'])
    dir_path = file_path.parent
    with open(os.path.join(dir_path,"train_param.json"), 'r') as f:
        param_dict = json.load(f) 
    params = TrainParam(**param_dict)
    params.batchsize = 8
    encoder = uvit.Unet1(params)
    ckpt = torch.load(args['ckpt'],map_location=torch.device('cpu'))
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()
    encoder.to(device)
    del ckpt
    image_mean = torch.tensor([0.5, 0.5, 0.5])
    image_std = torch.tensor([0.5, 0.5, 0.5])
    jnd_ = jnd.JND(preprocess=transforms.Normalize(-image_mean / image_std, 1 / image_std), 
                        postprocess=transforms.Normalize(image_mean, image_std)).to(device)
    
    # data preparation
    trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((args['input_res'],args['input_res'])),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])  
    dataset = build_dataset_I2V(datapath,trans=trans)
    dataloder = DataLoader(dataset,batch_size=params.batchsize,shuffle=False,num_workers=4,drop_last=True)
    encoded_path = os.path.join(save_path,"Encoded")
    os.makedirs(encoded_path, exist_ok=True)
    save_dict = {}
    save_dict['ckpt'] = args['ckpt']
    for image, id in dataloder:
        image = image.to(device)
        message_np = np.random.choice([0, 1], (image.shape[0], args['message']))
        for pos, i in enumerate(id):
            save_dict[i] = message_np[pos]
        message = torch.Tensor(message_np).to(device)
        encoded_images = encoder(image, message)

        # your post process
        if jnd_:
            encoded_images = params.scaling_i * image + params.scaling_w * encoded_images
            encoded_images = jnd_(image, encoded_images)
        for i in range(image.shape[0]):
            save_img(image[i:i+1],os.path.join(encoded_path, f'{id[i]}.png'))

    np.savez(os.path.join(save_path,"message_map.npz"), **save_dict)

@torch.no_grad
def encode_unfix(datapath, save_path, device, **args):
    #load your model
    file_path  = Path(args['ckpt'])
    dir_path = file_path.parent
    with open(os.path.join(dir_path,"train_param.json"), 'r') as f:
        param_dict = json.load(f) 
    params = TrainParam(**param_dict)
    params.batchsize = 1
    encoder = uvit.Unet1(params)
    ckpt = torch.load(args['ckpt'],map_location=torch.device('cpu'))
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()
    encoder.to(device)
    del ckpt
    image_mean = torch.tensor([0.5, 0.5, 0.5])
    image_std = torch.tensor([0.5, 0.5, 0.5])
    jnd_ = jnd.JND(preprocess=transforms.Normalize(-image_mean / image_std, 1 / image_std), 
                        postprocess=transforms.Normalize(image_mean, image_std)).to(device)
    
    # data preparation
    trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])  
    dataset = build_dataset_I2V(datapath,trans=trans)
    dataloder = DataLoader(dataset,batch_size=params.batchsize,shuffle=False,num_workers=4,drop_last=True)
    encoded_path = os.path.join(save_path,"Encoded")
    logger, log_path = setup_logger(save_path, "encode")
    os.makedirs(encoded_path, exist_ok=True)
    save_dict = {}
    save_dict['ckpt'] = args['ckpt']
    total_psnr = 0.0
    total_ssim = 0.0
    total_size = len(dataloder)
    for image, id in dataloder:
        image = image.to(device)
        message_np = np.random.choice([0, 1], (image.shape[0], args['message']))
        for pos, i in enumerate(id):
            save_dict[i] = message_np[pos]
        message = torch.Tensor(message_np).to(device)
        h,w = image.shape[2:]
        input_image = torch.nn.functional.interpolate(image, size=(args['input_res'],args['input_res']))
        encoded_images = encoder(input_image, message)

        # your post process
        encoded_images = torch.nn.functional.interpolate(encoded_images, size=(h,w))
        if jnd_:
            encoded_images = params.scaling_i * image + params.scaling_w * encoded_images
            encoded_images = jnd_(image, encoded_images)
        res = (encoded_images - image).clamp(-1, 1)
        encoded_images = 1.2*res.clamp(-1, 1) + image
        psnr_, ssim_, lpips_ = calculate_visual_metrics(encoded_images,image)
        total_psnr += psnr_
        total_ssim += ssim_
        for i in range(image.shape[0]):
            save_img(encoded_images[i:i+1],os.path.join(encoded_path, f'{id[i]}.png'))
    logger.info(f"AVG: EncOri PSNR {total_psnr/total_size}  SSIM{total_ssim/total_size} ")
    np.savez(os.path.join(save_path,"message_map.npz"), **save_dict)

@torch.no_grad
def decode_Videos(datapath, save_path, device, **args):
    message_map  = np.load(os.path.join(datapath, "message_map.npz"))
    
    #load your model
    file_path  = Path(message_map['ckpt'].item())
    dir_path = file_path.parent
    parent_dir = dir_path.parent
    with open(os.path.join(parent_dir,"train_param.json"), 'r') as f:
        param_dict = json.load(f) 
    params = TrainParam(**param_dict)
    decoder = convnext.ConvNextDecoder(params)
    ckpt = torch.load(message_map['ckpt'].item(),map_location=torch.device('cpu'))
    decoder.load_state_dict(ckpt["decoder"])
    decoder.eval()
    decoder.to(device)
    del ckpt
    sd_vae = load_VAE(device)
    trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((args['input_res'],args['input_res'])),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]) 
    logger, log_path = setup_logger(datapath, "decode")
    logger.info(message_map['ckpt'].item())
    filenames = os.listdir(os.path.join(datapath, "Videos"))

    acc_dict = {}
    for filename in filenames:
        video_names = os.listdir(os.path.join(datapath, "Videos", filename))
        avg_method_acc = 0.0
        all_acc = []
        for video_name in video_names:
            video = load_video(os.path.join(datapath,"Videos", filename, video_name))
            message = torch.Tensor([message_map[video_name.split('.')[0]]]).to(device)
            avg_Acc = []
            all_msg_t = []
            all_msg_bits = []
            for i, frame in enumerate(video):
                image = trans(frame).to(device)
                image = image.unsqueeze(0)
                decoded_tensor = decoder(image)
                all_msg_t.append(decoded_tensor)
                all_msg_bits.append(decoded_tensor.detach().cpu().numpy().round().clip(0, 1))
                avg_Acc.append(calculate_acc(message, decoded_tensor))
            # print(f"{filename} {video_name} : Acc: {sum(avg_Acc)/len(video)}    First Ten Frame {avg_Acc[:10]}   Flicker Index: {calculate_flicker_index(video)}")
            logger.info(f"{filename} {video_name} : Acc: {sum(avg_Acc)/len(video)}  Soft Vot Acc {calculate_acc(message,soft_vote(all_msg_t))} Hard Vot Acc {calculate_acc(message,torch.from_numpy(hard_vote(all_msg_bits)))}  First Ten Frame {avg_Acc[:10]}  " \
                          f"Flicker Index: {calculate_flicker_index(video)}  " \
                            f" SDVAE Cosine Similarity: {calculate_SDVAE_Cosine_Similarity(video, device, sd_vae)}   Wrapper Cosine Similarity: {calculate_Clip_Cosine_Similarity(video)}")
            avg_method_acc += sum(avg_Acc)/len(video)
            all_acc.append(avg_Acc)
        logger.info(f"{filename}  : Acc: {avg_method_acc/len(video_names)} ")
        acc_dict[filename] = np.array(all_acc)
    np.savez(os.path.join(datapath,"video_acc_map.npz"), **acc_dict)

@torch.no_grad
def decode_Videos_opticflow(datapath, save_path, device, **args):
    flow_model = TorchvisionRAFT()
    def compute_flow(ref: np.ndarray, target: np.ndarray, method: Literal['tvl1','farneback']='tvl1') -> np.ndarray:
        g0 = to_gray(ref)
        g1 = to_gray(target)
        if method == 'tvl1':
            try:
                tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
                flow = tvl1.calc(g0, g1, None)
                return flow
            except Exception:
                method = 'farneback'
        if method == 'farneback':
            flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 4, 21, 3, 7, 1.5, 0)
            return flow

    def warp_to_ref(ref: np.ndarray, target: np.ndarray, flow: np.ndarray) -> np.ndarray:
        h, w = ref.shape[:2]
        grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        map_x = grid_x + flow[...,0].astype(np.float32)
        map_y = grid_y + flow[...,1].astype(np.float32)
        warped = cv2.remap(target, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return warped

    def align_frames_to_reference(frames: List[np.ndarray], ref_index: int = 0, method: Literal['tvl1','farneback']='tvl1', flow_model: Optional[TorchvisionRAFT]=None) -> List[np.ndarray]:
        ref = frames[ref_index]
        aligned = []
        for i,f in enumerate(frames):
            if i == ref_index:
                aligned.append(f)
                continue
            if flow_model is not None:
                flow = flow_model.compute_flow(ref, f)
                warped = warp_to_ref(ref, f, flow)
            else:
                flow = compute_flow(ref,f,method)
                warped = warp_to_ref(ref,f,flow)
            aligned.append(warped)
        return aligned

    message_map  = np.load(os.path.join(datapath, "message_map.npz"))
    
    #load your model
    file_path  = Path(message_map['ckpt'].item())
    dir_path = file_path.parent
    parent_dir = dir_path.parent
    with open(os.path.join(parent_dir,"train_param.json"), 'r') as f:
        param_dict = json.load(f) 
    params = TrainParam(**param_dict)
    decoder = convnext.ConvNextDecoder(params)
    ckpt = torch.load(message_map['ckpt'].item(),map_location=torch.device('cpu'))
    decoder.load_state_dict(ckpt["decoder"])
    decoder.eval()
    decoder.to(device)
    del ckpt
    sd_vae = load_VAE(device)
    trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((args['input_res'],args['input_res'])),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]) 
    logger, log_path = setup_logger(datapath, "decode_optical")
    logger.info(message_map['ckpt'].item())
    filenames = os.listdir(os.path.join(datapath, "Videos"))
    os.makedirs(os.path.join(datapath, "Aligned"), exist_ok=True)
    acc_dict = {}
    for filename in filenames:
        video_names = os.listdir(os.path.join(datapath, "Videos", filename))
        avg_method_acc = 0.0
        avg_method_acc_vote = 0.0
        all_acc = []
        for video_name in video_names:
            os.makedirs(os.path.join(datapath, "Aligned",filename), exist_ok=True)
            video = load_video(os.path.join(datapath,"Videos", filename, video_name))
            message = torch.Tensor([message_map[video_name.split('.')[0]]]).to(device)
            h,w = video[0].size
            if h // 8!=0 or w // 8!=0:
                new_h = (h // 8) * 8
                new_w = (w // 8) * 8
                video = [frame.resize((new_w,new_h)) for frame in video]
                print(f"Resize to {new_h} {new_w}")
            np_videos = [np.array(frame) for frame in video]
            
            aligned_video = align_frames_to_reference(np_videos, ref_index=0, method='tvl1', flow_model=flow_model)
            video_op = [Image.fromarray(frame) for frame in aligned_video]
            export_to_video(video_op, os.path.join(datapath,"Aligned", filename, f"{video_name.split('.')[0]}_aligned.mp4"), fps=8)
            avg_Acc = []
            all_msg_t = []
            all_msg_bits = []
            for i, frame in enumerate(video_op):
                image = trans(frame).to(device)
                image = image.unsqueeze(0)
                decoded_tensor = decoder(image)
                all_msg_t.append(decoded_tensor)
                all_msg_bits.append(decoded_tensor.detach().cpu().numpy().round().clip(0, 1))
                avg_Acc.append(calculate_acc(message, decoded_tensor))
            logger.info(f"{filename} {video_name} : Acc: {sum(avg_Acc)/len(video)}  Soft Vot Acc {calculate_acc(message,soft_vote(all_msg_t))} Hard Vot Acc {calculate_acc(message,torch.from_numpy(hard_vote(all_msg_bits)))}  First Ten Frame {avg_Acc[:10]}  " \
                          f"Flicker Index: {calculate_flicker_index(video)}  " \
                            f" SDVAE Cosine Similarity: {calculate_SDVAE_Cosine_Similarity(video, device, sd_vae)}   Wrapper Cosine Similarity: {calculate_Clip_Cosine_Similarity(video)}")
            avg_method_acc += sum(avg_Acc)/len(video)
            avg_method_acc_vote += calculate_acc(message,torch.from_numpy(hard_vote(all_msg_bits)))
            all_acc.append(avg_Acc)
        logger.info(f"{filename}  : Acc: {avg_method_acc/len(video_names)} Vote Acc: {avg_method_acc_vote/len(video_names)} ")
        acc_dict[filename] = np.array(all_acc)
    np.savez(os.path.join(datapath,"video_acc_map_optical.npz"), **acc_dict)


    
if __name__=="__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    opt = NoiseOptions("video_config.yml")
    opt = opt.get_noise_opt()

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--datapath', type=str)
    parser.add_argument('--input_res', default=256, type=int)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--message', type=int, default=32)
    parser.add_argument('--test_num', type=int, default=10)
    parser.add_argument('--mode',  type=str, help="encode, I2V, decode")

    args = parser.parse_args()
    setup_seed(args.seed)  

    

    if args.mode == 'encode':
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = f"./Validation_Results/Video{timestamp}"
        os.makedirs(save_path, exist_ok=True)
        encode_unfix(args.datapath, save_path, device,
               ckpt=args.ckpt, input_res=args.input_res, message=args.message)


    elif args.mode == 'I2V':
        shutil.copy("video_config.yml", args.datapath)
        choices = opt['model']['Choice']['names']
        for choice in choices:
            setting = opt['model'][choice]
            resolution = choose_resolution(setting['resolution'])
            func = choose_test([choice])[0]
            func(input_image_path=os.path.join(args.datapath,"Encoded"), 
                 out_video_path=os.path.join(args.datapath,"Videos"),
                 resolution=resolution, guidance_scale=setting['guidance_scale'],
                 steps=setting['inference_steps'], fps=setting['fps'], 
                 quality=setting['quality'], test_num = args.test_num, device=device, prompt="data/video_prompts.json")

    elif args.mode == 'decode':
        save_path = None
        decode_Videos(args.datapath, save_path, device,
               ckpt=args.ckpt, input_res=args.input_res, message=args.message)
        
        decode_Videos_opticflow(args.datapath, save_path, device,
               ckpt=args.ckpt, input_res=args.input_res, message=args.message)


