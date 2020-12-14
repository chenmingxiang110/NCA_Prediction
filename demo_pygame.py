import pygame
import pygame.freetype

import os
import time
import pickle

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from lib.displayer2 import displayer
from lib.utils import mat_distance, softmax
from lib.NCCAModel2 import NCCAModel2, infer

with open('anchor_loc.pickle', 'rb') as handle:
    anchor_loc = pickle.load(handle)
anchor_loc = list(anchor_loc.keys())

def clear_screen(disp):
    disp.clear()
    btn_1 = pygame.draw.rect(disp.screen, (128,255,128), ( 15,bottom_y+15,30,30), 0)
    btn_2 = pygame.draw.rect(disp.screen, (255,255,128), ( 60,bottom_y+15,30,30), 0)
    btn_3 = pygame.draw.rect(disp.screen, (255,179, 51), (105,bottom_y+15,30,30), 0)
    btn_4 = pygame.draw.rect(disp.screen, (255,128,128), (150,bottom_y+15,30,30), 0)
    btn_0 = pygame.draw.rect(disp.screen, (0,0,0),       (195,bottom_y+15,105,30), 2)
    color_btns = [btn_0, btn_1, btn_2, btn_3, btn_4]

    btn_clear = pygame.draw.rect(disp.screen, (0,0,0), ( 15,bottom_y+60,75,30), 0)
    textsurface, _ = myfont.render('Clear', (255,255,255))
    disp.screen.blit(textsurface,(27,bottom_y+67))
    btn_size  = pygame.draw.rect(disp.screen, (0,0,0), (105,bottom_y+60,75,30), 0)
    textsurface, _ = myfont.render('Size '+str(pen_radius+1), (255,255,255))
    disp.screen.blit(textsurface,(115,bottom_y+67))
    btn_sim   = pygame.draw.rect(disp.screen, (0,0,0), (195,bottom_y+60,105,30), 0)
    textsurface, _ = myfont.render('Simulate', (255,255,255))
    disp.screen.blit(textsurface,(206,bottom_y+67))
    btn_map   = pygame.draw.rect(disp.screen, (0,0,0), (315,bottom_y+15,75,75), 2)
    textsurface, _ = myfont.render('Map '+str(valid_index), (0,0,0))
    disp.screen.blit(textsurface,(325,bottom_y+45))

    pygame.display.update()

pen_radius = 2
max_pen_radius = 8
pix_size = 5
map_size = (80,80)

pen_index = 1
valid_index = 0
valid_masks = np.load("valid_masks.npy")
print(anchor_loc[valid_index])

####################################################
# Model Init

device_name = "cpu"
DEVICE = torch.device(device_name)
# model_path = "models/ncca_softmax_multi_traffic.pth"
model_path = "models/ncca_softmax_traffic.pth"
CHANNEL_N = 16
ALPHA_CHANNEL = 4
N_STEPS = 128
CELL_FIRE_RATE = 0.5

my_model = NCCAModel2(CHANNEL_N, ALPHA_CHANNEL, CELL_FIRE_RATE, DEVICE).to(DEVICE)
if device_name == "cpu":
    my_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
else:
    my_model.load_state_dict(torch.load(model_path))
for param in my_model.parameters():
    param.requires_grad = False

####################################################

_rows = np.arange(map_size[0]).repeat(map_size[1]).reshape(map_size)
_cols = np.arange(map_size[1]).reshape([1,-1]).repeat(map_size[0],axis=0)
_map_pos = np.array([_rows,_cols]).transpose([1,2,0])

_map = np.zeros([map_size[0], map_size[1], CHANNEL_N]) * np.rot90(valid_masks[valid_index],1)
disp = displayer(map_size, pix_size, ALPHA_CHANNEL)
disp.draw_all(_map, np.rot90(valid_masks[valid_index],1))

bottom_y = _map_pos.shape[0]*pix_size

myfont = pygame.freetype.SysFont('Comic Sans MS', 20)

btn_1 = pygame.draw.rect(disp.screen, (128,255,128), ( 15,bottom_y+15,30,30), 0)
btn_2 = pygame.draw.rect(disp.screen, (255,255,128), ( 60,bottom_y+15,30,30), 0)
btn_3 = pygame.draw.rect(disp.screen, (255,179, 51), (105,bottom_y+15,30,30), 0)
btn_4 = pygame.draw.rect(disp.screen, (255,128,128), (150,bottom_y+15,30,30), 0)
btn_0 = pygame.draw.rect(disp.screen, (0,0,0),       (195,bottom_y+15,105,30), 2)
color_btns = [btn_0, btn_1, btn_2, btn_3, btn_4]

btn_clear = pygame.draw.rect(disp.screen, (0,0,0), ( 15,bottom_y+60,75,30), 0)
textsurface, _ = myfont.render('Clear', (255,255,255))
disp.screen.blit(textsurface,(27,bottom_y+67))
btn_size  = pygame.draw.rect(disp.screen, (0,0,0), (105,bottom_y+60,75,30), 0)
textsurface, _ = myfont.render('Size '+str(pen_radius+1), (255,255,255))
disp.screen.blit(textsurface,(115,bottom_y+67))
btn_sim   = pygame.draw.rect(disp.screen, (0,0,0), (195,bottom_y+60,105,30), 0)
textsurface, _ = myfont.render('Simulate', (255,255,255))
disp.screen.blit(textsurface,(206,bottom_y+67))
btn_map   = pygame.draw.rect(disp.screen, (0,0,0), (315,bottom_y+15,75,75), 2)
textsurface, _ = myfont.render('Map '+str(valid_index), (0,0,0))
disp.screen.blit(textsurface,(325,bottom_y+45))

pygame.display.update()

isMouseDown = False
running = True
history = []
while running:
    if len(history)>0:
        print(len(history))
        new_map = np.rot90(history.pop()[0],1)
        if len(history)==0: print("Simulation Complete")
        clear_screen(disp)
        disp.draw_all(new_map, np.rot90(valid_masks[valid_index],1))
        pygame.event.pump()
        continue

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                isMouseDown = True

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                isMouseDown = False

            for i_btn, c_btn in enumerate(color_btns):
                if c_btn.collidepoint(event.pos):
                    pen_index = i_btn
                    clear_screen(disp)
                    disp.draw_all(_map, np.rot90(valid_masks[valid_index],1))
            if btn_clear.collidepoint(event.pos):
                _map = np.zeros(_map.shape) * np.rot90(valid_masks[valid_index],1)
                disp.draw_all(_map, np.rot90(valid_masks[valid_index],1))
            if btn_size.collidepoint(event.pos):
                pen_radius+=2
                if pen_radius>max_pen_radius: pen_radius=0
                clear_screen(disp)
                disp.draw_all(_map, np.rot90(valid_masks[valid_index],1))
            if btn_sim.collidepoint(event.pos):
                x = torch.from_numpy(np.expand_dims(np.rot90(_map,3), 0).astype(np.float32)).to(DEVICE)
                induction = torch.from_numpy(np.expand_dims(np.rot90(_map,3), 0).astype(np.float32)).to(DEVICE)
                valid_mask_t = torch.from_numpy(np.expand_dims(valid_masks[valid_index], 0).astype(np.float32)).to(DEVICE)
                calibration_map = np.expand_dims(np.expand_dims(np.rot90(_map,3)[...,ALPHA_CHANNEL], 0), -1).astype(np.float32)
                calibration_map = torch.from_numpy(calibration_map).to(DEVICE)
                x, history = infer(my_model, x, ALPHA_CHANNEL, induction, valid_mask_t, calibration_map, N_STEPS)
            if btn_map.collidepoint(event.pos):
                valid_index+=1
                if valid_index>9: valid_index=0
                print(anchor_loc[valid_index])
                _map = np.zeros(_map.shape) * np.rot90(valid_masks[valid_index],1)
                clear_screen(disp)
                disp.draw_all(_map, np.rot90(valid_masks[valid_index],1))

    if isMouseDown:
        try:
            mouse_pos = np.array([int(event.pos[1]/pix_size), int(event.pos[0]/pix_size)])
            if mouse_pos[0]<_map_pos.shape[0]:
                draw = (mat_distance(_map_pos, mouse_pos)<=pen_radius).reshape([map_size[0],map_size[1],1]).astype(float)
                changed = np.argwhere(draw>0)
                changed = changed[...,:2]
                for i,j in changed:
                    values = np.zeros(CHANNEL_N)
                    if pen_index>0:
                        one_hot = np.zeros(ALPHA_CHANNEL+1)
                        one_hot[-1] = 1
                        one_hot[pen_index-1] = 1
                        values = np.concatenate((one_hot, np.random.random(CHANNEL_N-ALPHA_CHANNEL-1)))
                    _map[i,j] = values
                disp.draw(_map, np.rot90(valid_masks[valid_index],1), changed)
        except AttributeError:
            pass
