import os
import csv
import time
import torch
import logging
import torchvision
import random
import numpy as np
from kornia.metrics import psnr,ssim
from Noise_pool import Noise_pool
import cv2
from PIL import Image
from diffusers import AutoencoderKL
from transformers import CLIPVisionModel, AutoProcessor
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
import lpips

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
lpips = lpips.LPIPS(net='alex').to(device)
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.benchmark = False
     torch.backends.cudnn.deterministic = True

def save_img(origin_img,filename):
    img = origin_img[:,:,:,:].cpu()
    img = (img + 1) / 2
    torchvision.utils.save_image(img,filename,normalize=False)

def calculate_visual_metrics(encoded_images, images):
    psnr_ = psnr(((encoded_images+1)/2).clamp(0,1),((images+1)/2).clamp(0,1),1.).item()
    ssim_ = ssim(((encoded_images+1)/2).clamp(0,1),((images+1)/2).clamp(0,1),5).mean().item() 
    lpips_ = lpips(((encoded_images+1)/2).clamp(0,1),((images+1)/2).clamp(0,1)).mean().item()
    return psnr_, ssim_, lpips_

def calculate_flicker_index(frames):
    frames = [np.array(f) for f in frames]
    stack = np.stack(frames, axis=0).astype(np.float32)
    return np.mean(np.std(stack, axis=0))

def calculate_acc(gt_message, decoded_tensor):
    decoded_rounded = decoded_tensor.detach().cpu().numpy().round().clip(0, 1)
    bitwise_avg_err = np.sum(np.abs(decoded_rounded - gt_message.detach().cpu().numpy())) / (
    gt_message.shape[0] * gt_message.shape[1])
    acc = 1-bitwise_avg_err
    return acc

def calculte_frame_SSIM(frames):
    frames = [np.array(f) for f in frames]
    stack = np.stack(frames, axis=0).astype(np.float32)
    ssim_list = []
    for i in range(1, len(stack)):
        ssim_ = ssim(torch.from_numpy(stack[i-1:i]), torch.from_numpy(stack[i:i+1]), 5).mean().item()
        ssim_list.append(ssim_)
    return np.mean(ssim_list)

def calculate_WrapError(frames):
    frames = [np.array(f) for f in frames]
    errs = []
    for i in range(1, len(frames)):
        prev, curr = frames[i-1], frames[i]
        prev_gray = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)
        h, w = flow.shape[:2]
        flow_map = np.column_stack((np.repeat(np.arange(h), w),
                                    np.tile(np.arange(w), h))).reshape(h, w, 2).astype(np.float32)
        warp_coords = flow_map + flow
        warped = cv2.remap(prev, warp_coords[:,:,1], warp_coords[:,:,0], interpolation=cv2.INTER_LINEAR)
        err = np.mean(np.abs(warped.astype(np.float32) - curr.astype(np.float32)))
        errs.append(err)
    return np.mean(errs)

def load_clip(device):
    model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    model.to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

def load_VAE(device):
    model = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    model.to(device)
    model.eval()
    return model

def soft_vote(tensors):
    if not tensors:
        return None
    stacked = torch.stack(tensors, dim=0)
    return torch.mean(stacked, dim=0)

def hard_vote(bits):
    stack = np.stack(bits, axis=0).astype(np.int32)
    votes = np.sum(stack, axis=0)
    return (votes >= (len(bits) / 2)).astype(np.uint8)

@torch.no_grad
def calculate_Clip_Cosine_Similarity(frames, model=None, processor=None):
    frames = [np.array(f) for f in frames]
    stack = np.stack(frames, axis=0).astype(np.float32)
    cos_sim_list = []
    for i in range(1, len(stack)):
        prev, curr = stack[i-1], stack[i]
        prev_norm = np.linalg.norm(prev)
        curr_norm = np.linalg.norm(curr)
        if prev_norm == 0 or curr_norm == 0:
            cos_sim = 0
        else:
            cos_sim = np.dot(prev.flatten(), curr.flatten()) / (prev_norm * curr_norm)
        cos_sim_list.append(cos_sim)
    return np.mean(cos_sim_list)

@torch.no_grad
def calculate_SDVAE_Cosine_Similarity(frames, device, model=None):
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # [-1,1]
    ])
    all_latents = []
    cos_sim_list = []
    bs = 4
    for i in range(0,len(frames),bs):
        batch_frames = frames[i:i+bs] if i+bs <= len(frames) else frames[i:]
        batch_tensor = torch.stack([preprocess(f) for f in batch_frames]).to(device)
        latents = model.encode(batch_tensor).latent_dist.mean
        latents = latents * 0.18215
        latents = latents.reshape(latents.size(0), -1)
        all_latents.append(latents.cpu())
    all_latents = torch.cat(all_latents, dim=0)
    for i in range(len(all_latents) - 1):
        sim = F.cosine_similarity(all_latents[i].unsqueeze(0), all_latents[i+1].unsqueeze(0)).item()
        cos_sim_list.append(sim)
    return np.mean(cos_sim_list)

@torch.no_grad
def test_fix_resolution(encoder,decoder, noise_opt, dataloader, save_path, device, **args):
    noise_model = Noise_pool(noise_opt, device).to(device)
    for choice in noise_opt['noise']['All']['names']:
        total_psnr = 0.0
        total_ssim = 0.0
        total_psnr_noise = 0.0
        total_ssim_noise = 0.0
        avg_acc = 0.0
        total_size = 0
        print(f"Noise Choice : {choice}")
        for i, image in enumerate(dataloader):
            image = image.to(device)
            message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], args['message']))).to(device)

            encoded_images = encoder(image, message)
            # your own postprocess
            if args['jnd']:
                encoded_images = encoder.scaling_i * image + encoder.scaling_w * encoded_images
                encoded_images = args['jnd'](image, encoded_images)

            noised_images = noise_model(encoded_images, image, choice)
            to_save = noised_images.clone()
            decoded_tensor = decoder(noised_images)
            psnr_, ssim_, lpips_ = calculate_visual_metrics(encoded_images,image)
            acc = calculate_acc(message, decoded_tensor)
            total_psnr += psnr_
            total_ssim += ssim_
            psnr_, ssim_ = calculate_visual_metrics(encoded_images,to_save)
            total_psnr_noise += psnr_
            total_ssim_noise += ssim_
            avg_acc += acc
            total_size += 1
        save_img(torch.concat([image, encoded_images, 10*(encoded_images-image).clamp(-1,1),noised_images],dim=0), os.path.join(save_path,f"{choice}_{i}.png"))
        print(f"AVG: EncOri PSNR {total_psnr/total_size}  SSIM{total_ssim/total_size}  ACC {avg_acc/total_size} ")
        print(f"AVG: EncNoise PSNR {total_psnr_noise/total_size}  SSIM{total_ssim_noise/total_size}  ")
    return

@torch.no_grad
def test_unfix_resolution(encoder,decoder, noise_opt, dataloader, save_path, device, **args):
    noise_model = Noise_pool(noise_opt, device).to(device)
    for choice in noise_opt['noise']['All']['names']:
        total_psnr = 0.0
        total_ssim = 0.0
        total_lpips = 0.0
        total_psnr_noise = 0.0
        total_ssim_noise = 0.0
        avg_acc = 0.0
        total_size = 0
        print(f"Noise Choice : {choice}")
        for i, image in enumerate(dataloader):
            image = image.to(device)
            h,w = image.shape[2:]
            input_image = torch.nn.functional.interpolate(image,size=(args['resolution'],args['resolution']))
            message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], args['message']))).to(device)

            encoded_images = encoder(input_image, message)
            # your own postprocess
            encoded_images = torch.nn.functional.interpolate(encoded_images, size=(h,w))
            f_encoded_images = encoded_images.clone()
            if args['jnd']:
                encoded_images = encoder.scaling_i * image + encoder.scaling_w * encoded_images
                encoded_images = args['jnd'](image, encoded_images)
            res = (encoded_images - image).clamp(-1, 1)
            f_encoded_images = res.clone()
            encoded_images = 1.2*res.clamp(-1, 1) + image
            
            noised_images = noise_model(encoded_images, image, choice)
            to_save = noised_images.clone()
            noised_images = torch.nn.functional.interpolate(noised_images,size=(args['resolution'],args['resolution']))
            
            decoded_tensor = decoder(noised_images)
            psnr_, ssim_, lpips_= calculate_visual_metrics(encoded_images,image)
            acc = calculate_acc(message, decoded_tensor)
            total_psnr += psnr_
            total_ssim += ssim_
            total_lpips += lpips_

            avg_acc += acc
            total_size += 1

        save_img(image, os.path.join(save_path,f"o_{choice}_{i}.png"))
        save_img(f_encoded_images, os.path.join(save_path,f"fe_{choice}_{i}.png"))
        save_img(encoded_images, os.path.join(save_path,f"e_{choice}_{i}.png"))
        print(f"AVG: EncOri PSNR {total_psnr/total_size}  SSIM{total_ssim/total_size} LPIPS {total_lpips/total_size}  ACC {avg_acc/total_size} ")

    return

def setup_logger(datapath, name):
    os.makedirs(datapath, exist_ok=True)
    log_path = os.path.join(datapath, f"{name}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()   # 避免重复添加 handler

    # 文件 handler
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)

    # 控制台 handler（会被 Slurm 收集到 slurm-xxx.out）
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger, log_path

def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def ensure_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)

class TorchvisionRAFT:
    def __init__(self, device="cuda"):
        weights = Raft_Large_Weights.DEFAULT
        self.model = raft_large(weights=weights).to(device)
        self.model = self.model.eval()
        self.transforms = weights.transforms()
        self.device = device

    
    def np_to_pil(self,img):
        if isinstance(img, np.ndarray):
            return Image.fromarray(img.astype(np.uint8))
        return img

    @torch.inference_mode()
    def compute_flow(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        pil_frame1 = self.np_to_pil(frame1)
        pil_frame2 = self.np_to_pil(frame2)
        batch = self.transforms(pil_frame1, pil_frame2)
        flow = self.model(batch[0].unsqueeze(0).to(self.device), batch[1].unsqueeze(0).to(self.device))
        flow = flow[-1] 

        H, W, _ = frame1.shape
        flow = F.interpolate(flow, size=(H, W), mode="bilinear", align_corners=False)
        flow = flow[0].permute(1, 2, 0).cpu().numpy()
        # flow = flow[0].permute(1, 2, 0)
        return flow

class SEA_RAFT:
    def __init__(self, device="cuda"):
        weights = Raft_Large_Weights.DEFAULT
        self.model = raft_large(weights=weights).to(device)
        self.model = self.model.eval()
        self.transforms = weights.transforms()
        self.device = device

