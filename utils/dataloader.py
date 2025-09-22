from PIL import Image
import os.path as p
import os
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,datasets
import torch
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class GetImageJPG(Dataset):
    def __init__(self,root, trans=None, convert_mode="RGB") -> None:
        self.trans = transforms.Compose([transforms.ToTensor()]) if trans is None else trans
        self.bg_path = p.join(root,"%s.jpg")
        self.root = root
        self.bg_ids = []
        self.convert_mode = convert_mode
        for file in os.listdir(root):
            self.bg_ids.append(file.split(".")[0])

    
    def __getitem__(self, index):
        bg = self.pull_item(index)
        return bg

    def __len__(self):
        return len(self.bg_ids) 
    
    def pull_item(self , index):

        bg_id = self.bg_ids[index]
        bg = Image.open(self.bg_path%bg_id).convert(self.convert_mode)
        bg = self.trans(bg)
        bg = bg * 2. - 1.
        return bg
            # return torch.zeros((3, 256, 256))  # 返回一个空白图片，防止 dataloader 出错

class GetImagePNG(Dataset):
    def __init__(self,root, trans=None) -> None:
        self.trans = transforms.Compose([transforms.ToTensor()]) if trans is None else trans
        self.bg_path = p.join(root,"%s.png")
        self.root = root
        self.bg_ids = []
        for file in os.listdir(root):
            self.bg_ids.append(file.split(".")[0])
    
    def __getitem__(self, index):
        bg = self.pull_item(index)
        return bg

    def __len__(self):
        return len(self.bg_ids) 
    
    def pull_item(self , index):
        bg_id = self.bg_ids[index]
        bg = Image.open(self.bg_path%bg_id)
        bg = self.trans(bg)
        # bg = bg * 2. - 1.
        return bg

class GetImageI2V(Dataset):
    def __init__(self,root, trans=None) -> None:
        self.trans = transforms.Compose([transforms.ToTensor()]) if trans is None else trans
        self.bg_path = p.join(root,"%s.png")
        self.root = root
        self.bg_ids = []
        for file in os.listdir(root):
            self.bg_ids.append(file.split(".")[0])
    
    def __getitem__(self, index):
        bg, bg_id = self.pull_item(index)
        return bg, bg_id

    def __len__(self):
        return len(self.bg_ids) 
    
    def pull_item(self , index):
        bg_id = self.bg_ids[index]
        bg = Image.open(self.bg_path%bg_id)
        bg = self.trans(bg)
        # bg = bg * 2. - 1.
        return bg, bg_id

def build_dataset(root, trans=None):
    dataset = GetImagePNG(root,trans=trans)
    return dataset

def build_dataset_I2V(root, trans=None):
    dataset = GetImageI2V(root,trans=trans)
    return dataset

def build_dataset_MSCOCO(root, trans=None):
    dataset = datasets.ImageFolder(root, trans)
    return dataset

class GetBackGround(Dataset):
    def __init__(self,root) -> None:
        
        self.trans = transforms.Compose([transforms.ToTensor(),transforms.Resize([1000,1000])])
        self.bg_path = p.join(root,"backgrounds","%s.jpg")
        self.root = root
        self.bg_ids = []
        for file in os.listdir(root+"/backgrounds"):
            self.bg_ids.append(file.strip(".jpg"))
    
    def __getitem__(self, index):
        bg = self.pull_item(index)
        return bg

    def __len__(self):
        return len(self.bg_ids) 
    
    def pull_item(self , index):
        bg_id = self.bg_ids[index]
        bg = Image.open(self.bg_path%bg_id)
        bg = self.trans(bg)
        bg = bg * 2. - 1.
        return bg

class GetImg_Mask(Dataset):
    def __init__(self,root) -> None:
        self.trans = transforms.Compose([transforms.ToTensor()])
        self.img_path = p.join(root,"images","%s.jpg")
        self.mask_path = p.join(root,"masks","%s.png")
        self.root = root
        self.im_msk_ids = []
        for file in os.listdir(root+"/images"):
            self.im_msk_ids.append(file.strip(".jpg"))
    
    def __getitem__(self, index):
        img , mask = self.pull_item(index)
        return img,mask

    def __len__(self):
        return len(self.im_msk_ids)
    
    def pull_item(self,index):
        im_id = self.im_msk_ids[index]
        img = Image.open(self.img_path%im_id)
        mask = Image.open(self.mask_path%im_id)

        img = self.trans(img)
        mask = self.trans(mask)
        img = img * 2. - 1.
        return img,mask

def dataloader_background(root, batch_size):
    data_loader = DataLoader(
        GetBackGround(root),
        batch_size=batch_size, shuffle=True 
    )
    return data_loader

def dataloader_img_mask(root, batch_size):
    data_loader = DataLoader(
        GetImg_Mask(root),
        batch_size=batch_size, shuffle=True 
    )
    return data_loader