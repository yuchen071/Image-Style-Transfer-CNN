# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 14:47:08 2020

@author: Eric
"""
import os
import re
import cv2
import torch
from torchvision import transforms

from src import utils
from src.transformer_net import TransformerNet
from tkinter.filedialog import askopenfilename

#%% Global Parameters
MODEL_PATH = askopenfilename(title="Select model file", filetypes=[('model path', '*.pth')], initialdir=os.getcwd())
if not MODEL_PATH: raise ValueError("No model selected")
SCALE = 1

#%%
def main():
    with torch.no_grad():
        
        # model stuff
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        content_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
        transformerNet = TransformerNet()
        state_dict = torch.load(MODEL_PATH)['state_dict']
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        transformerNet.load_state_dict(state_dict)
        transformerNet.to(device)
    
        # camera stuff
        # cv2.namedWindow("preview")
        cv2.namedWindow("output")
        vc = cv2.VideoCapture(0)  
        if vc.isOpened():
            rval, frame = vc.read()
        else:
            rval = False
        
        # main loop
        while rval:
            rval, frame = vc.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            content_image = utils.load_frame(frame, scale=SCALE)
            content_image = content_transform(content_image)
            content_image = content_image.unsqueeze(0).to(device)
        
            output = transformerNet(content_image).cpu()
            out_img = utils.read_image(output[0])
            
            # cv2.imshow("preview", frame)
            cv2.imshow("output", out_img)
        
            key = cv2.waitKey(20)
            if key == 27: # exit on ESC
                break
        vc.release()
        # cv2.destroyWindow("preview")
        cv2.destroyWindow("output")
        
        if str(device) == 'cuda':
            torch.cuda.empty_cache()
            
if __name__ == "__main__":
    main()