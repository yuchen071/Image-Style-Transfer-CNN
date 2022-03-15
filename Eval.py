# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 22:26:34 2020

@author: Eric
"""
import os
import re
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

from src import utils
from src.transformer_net import TransformerNet
from tkinter.filedialog import askopenfilename

#%% Global params
MODEL_PATH = askopenfilename(title="Select model file", filetypes=[('model path', '*.pth')], initialdir=os.getcwd())
if not MODEL_PATH: raise ValueError("No model selected")
TEST_IMG_PATH = askopenfilename(title="Select Target Image", filetypes=[('Image files', '*.jpg *.png')], initialdir=os.getcwd())
if not TEST_IMG_PATH: raise ValueError("No target image selected")
TEST_SCALE = 1
OUTPUT_DIR = "results"

try:
    style_name = torch.load(MODEL_PATH)['style name']
except:
    style_name = "output"
test_img_name, _ = os.path.splitext(os.path.basename(TEST_IMG_PATH))
out_img_name = f"{test_img_name}_{style_name}.jpg"

#%%
def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    
    content_image = utils.load_image(TEST_IMG_PATH, scale=TEST_SCALE)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        transformerNet = TransformerNet()
        state_dict = torch.load(MODEL_PATH)['state_dict']
        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        transformerNet.load_state_dict(state_dict)
        transformerNet.to(device)
        
        output = transformerNet(content_image).cpu()
        
        utils.check_path(OUTPUT_DIR)
        utils.save_image(os.path.join(OUTPUT_DIR, out_img_name), output[0])
        
        if str(device) == 'cuda':
            torch.cuda.empty_cache()
        
def loss_plot():
    content_loss = torch.load(MODEL_PATH)['content_loss']
    style_loss = torch.load(MODEL_PATH)['style_loss']
    content_weight = torch.load(MODEL_PATH)['content_weight']
    style_weight = torch.load(MODEL_PATH)['style_weight']
    
    plt.figure(0)
    plt.plot(content_loss, label="content loss")
    plt.plot(style_loss, label="style loss")
    plt.legend()
    title_str = f"Loss, cw = {content_weight}, sw = {style_weight}"
    # title_str = f"Learning rate = 1e-4"
    plt.title(title_str)
    plt.xlabel("iterations")
    plt.show()
    
#%%
if __name__ == "__main__":
    evaluate()
    loss_plot()