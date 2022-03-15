# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 13:42:16 2020

@author: Eric
"""
import os
from tkinter.filedialog import askopenfilename, askdirectory
from datetime import datetime as dt

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm

from src import utils
from src.transformer_net import TransformerNet
from src.vgg import Vgg_after_relu
# from src.vgg import Vgg_before_relu

#%% Global params
DATASET_DIR = askdirectory(title="Select COCO dataset directory")
if not DATASET_DIR: raise ValueError("No dataset directory selected")
STYLE_IMG = askopenfilename(title="Select Style Image", filetypes=[('Image files', '*.jpg *.png')], initialdir=os.getcwd())
if not STYLE_IMG: raise ValueError("No style image selected")
TEST_IMG = askopenfilename(title="Select Target Image", filetypes=[('Image files', '*.jpg *.png')], initialdir=os.getcwd())
if not TEST_IMG: raise ValueError("No target image selected")
STYLE_SCALE = 1

MODEL_FOLDER = "model"
# continue training with selected model.pth, set to None to disable
CONTINUE = None
# CONTINUE = askopenfilename(title="Select model file", filetypes=[('model path', '*.pth')], initialdir=os.getcwd())
if not CONTINUE: CONTINUE = None

EPOCHS = 1
LR = 1e-3
BATCH_SIZE = 4
TRAIN_IMG_SIZE = 256
CONTENT_WEIGHT = 1e0
STYLE_WEIGHT = 1e6
CHKPT_FREQ = 1000 # set to 0 to disable checkpoints
TEST_FREQ = 2000 # set to 0 to disable test images

TEST_IMG_NAME, _ = os.path.splitext(os.path.basename(TEST_IMG))
STYLE_IMG_NAME, _ = os.path.splitext(os.path.basename(STYLE_IMG))
MODEL_SAVE_FOLDER = os.path.join(MODEL_FOLDER, STYLE_IMG_NAME)
TEST_SAVE_FOLDER = os.path.join(MODEL_FOLDER, STYLE_IMG_NAME, "test images")

#%%
def train():
    # make model dir if not exist
    utils.check_path(MODEL_SAVE_FOLDER)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if TEST_FREQ != 0:
        utils.check_path(TEST_SAVE_FOLDER)
        
        test_image = utils.load_image(TEST_IMG)
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
        test_image = test_transform(test_image)
        test_image = test_image.unsqueeze(0).to(device)
    
    # load dataset
    transform = transforms.Compose([
        transforms.Resize(TRAIN_IMG_SIZE),
        transforms.CenterCrop(TRAIN_IMG_SIZE),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    train_dataset = datasets.ImageFolder(DATASET_DIR, transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    
    # load training network
    transformerNet = TransformerNet().to(device).train()
    # transformerNet = UNet().to(device).train()
    optimizer = Adam(transformerNet.parameters(), LR)
    mse_loss = torch.nn.MSELoss()
    
    # load previous state if enabled
    if CONTINUE is not None:
        state_dict = torch.load(CONTINUE)
        transformerNet.load_state_dict(state_dict['state_dict'])
        optimizer.load_state_dict(state_dict['optimizer'])
        content_loss_history = state_dict['content_loss']
        style_loss_history = state_dict['style_loss']
    else:
        content_loss_history = []
        style_loss_history = []
    
    # extract target style img features via style network (VGG)
    vgg = Vgg_after_relu(requires_grad=False).to(device)
    # vgg = Vgg_before_relu(requires_grad=False).to(device)
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    style = utils.load_image(STYLE_IMG, scale=STYLE_SCALE)
    style = style_transform(style)
    style = style.repeat(BATCH_SIZE, 1, 1, 1).to(device)

    style_features = vgg(utils.normalize_batch(style))
    style_gram = [utils.gram_matrix(x) for x in style_features]
    
    # train
    time_start = dt.now()
    torch.autograd.set_detect_anomaly(True) # for debugging
    
    for epoch in range(EPOCHS):
        
        # tqdm description (epoch number)
        t = tqdm(train_loader)
        t_desc = "Epoch %2d/%d" % (epoch + 1, EPOCHS)
        t.set_description(t_desc)
        
        for batch_id, (x_original, _) in enumerate(t):                 
            # feed into train network
            x_original = x_original.to(device)
            x_stylized = transformerNet(x_original)
            
            # extract features via vgg net
            x_original_features = vgg(utils.normalize_batch(x_original))
            x_stylized_features = vgg(utils.normalize_batch(x_stylized))
            
            # content loss calculation
            x_original_rep = x_original_features.relu2_2
            x_stylized_rep = x_stylized_features.relu2_2
            content_loss = CONTENT_WEIGHT * mse_loss(x_original_rep, x_stylized_rep)
            
            # style loss calculation
            style_loss = 0.
            x_stylized_gram = [utils.gram_matrix(x) for x in x_stylized_features]
            for x_gm, target_gm in zip(x_stylized_gram, style_gram):
                style_loss += mse_loss(x_gm, target_gm[:len(x_original), :, :])
            style_loss /= len(style_gram)
            style_loss *= STYLE_WEIGHT
            
            # combine losses and backpropagation
            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # append loss history
            content_loss_history.append(content_loss.item())
            style_loss_history.append(style_loss.item())
            
            # save checkpoint model
            if CHKPT_FREQ != 0 and (batch_id + 1) % CHKPT_FREQ == 0:
                transformerNet.eval().cpu()
                chkpt_state = {
                    'style name': STYLE_IMG_NAME,
                    'epoch': epoch,
                    'content_weight': CONTENT_WEIGHT,
                    'style_weight': STYLE_WEIGHT,
                    'state_dict': transformerNet.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'content_loss': content_loss_history,
                    'style_loss': style_loss_history
                    }
                chkpt_filename = f"chkpt_epoch_{epoch + 1}_batch_{batch_id + 1}_cw_{CONTENT_WEIGHT}_sw_{STYLE_WEIGHT}.pth"
                torch.save(chkpt_state, os.path.join(MODEL_SAVE_FOLDER, chkpt_filename))
                transformerNet.to(device).train()
                
            # update tqdm postfix (loss)
            t_postfix_content = "%.2f" % (content_loss.item())
            t_postfix_style = "%.2f" % (style_loss.item())
            t.set_postfix({'Content loss': t_postfix_content, 'Style loss': t_postfix_style})
            t.update(0)
            
            if TEST_FREQ != 0 and (batch_id + 1) % TEST_FREQ == 0:
                with torch.no_grad():
                    out = transformerNet(test_image).cpu()
                    out_filename = f"test_epoch_{epoch + 1}_batch_{batch_id + 1}_cw_{CONTENT_WEIGHT}_sw_{STYLE_WEIGHT}.jpg"
                    utils.save_image(os.path.join(TEST_SAVE_FOLDER, out_filename), out[0])
            
    # save model
    transformerNet.eval().cpu()
    model_state = {
        'style name': STYLE_IMG_NAME,
        'epoch': epoch,
        'content_weight': CONTENT_WEIGHT,
        'style_weight': STYLE_WEIGHT,
        'state_dict': transformerNet.state_dict(),
        'optimizer': optimizer.state_dict(),
        'content_loss': content_loss_history,
        'style_loss': style_loss_history
        }
    model_filename = f"{STYLE_IMG_NAME}_epoch_{EPOCHS}_cw_{CONTENT_WEIGHT}_sw_{STYLE_WEIGHT}.pth"
    save_path = os.path.join(MODEL_SAVE_FOLDER, model_filename)
    torch.save(model_state, save_path)
    
    print("\n\nTraining Complete!\nModel saved at", save_path)
    
    # Elapsed time
    elapsed = dt.now() -  time_start
    print("Elapsed time: %02d:%02d:%02d" % (elapsed.seconds // 3600, elapsed.seconds // 60 % 60, elapsed.seconds % 60))
    
    if str(device) == 'cuda':
        torch.cuda.empty_cache()
            
#%%  
if __name__ == "__main__":
    train()
    
