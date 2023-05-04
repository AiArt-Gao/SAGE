import argparse
import math
import glob
import random

import cv2
import numpy as np
import sys
import os

import torch
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
from PIL import Image
from torch_ema import ExponentialMovingAverage
from generators import generators
from siren import siren
from torchvision import transforms

import curriculums

device = torch.device('cuda')


COLOR_MAP = {
        0: [0, 0, 0],
        1: [204, 0, 0],
        2: [76, 153, 0],
        3: [204, 204, 0],
        4: [51, 51, 255],
        5: [204, 0, 204],
        6: [0, 255, 255],
        7: [255, 204, 204],
        8: [102, 51, 0],
        9: [255, 0, 0],
        10: [102, 204, 0],
        11: [255, 255, 0],
        12: [0, 0, 153],
        13: [0, 0, 204],
        14: [255, 51, 153],
        15: [0, 204, 204],
        16: [0, 51, 0],
        17: [255, 153, 51],
        18: [0, 204, 0]
    }


def mask2color(masks):

    masks = torch.argmax(masks, dim=1).float()
    sample_mask = torch.zeros((masks.shape[0], masks.shape[1], masks.shape[2], 3), dtype=torch.float)
    for key in COLOR_MAP.keys():
        sample_mask[masks == key] = torch.tensor(COLOR_MAP[key], dtype=torch.float)
    sample_mask = sample_mask.permute(0, 3, 1, 2)
    return sample_mask


def getParsing(path):
    image = Image.open(path).convert('RGB')
    transform = transforms.ToTensor()
    image = transform(image)
    image = image.permute(1, 2, 0).reshape((256 * 256, 3))
    semantic_map = torch.zeros((256 * 256, 19))

    for index, pixel in enumerate(image):
        channel = 0
        for key, value in COLOR_MAP.items():
            if value == list(255 * pixel.data):
                # print("ok")
                # if key != 13:
                channel = key
        for cc in range(0, 19):
            if cc == channel:
                semantic_map[index, channel] = float(np.random.random(1)+0.5)
            else:
                semantic_map[index, cc] = float(-np.random.random(1) - 0.6)

    semantic_map = semantic_map.reshape((256, 256, 19))
    semantic_map = semantic_map.permute(2, 0, 1)
    semantic_map = semantic_map.unsqueeze(0)
    return semantic_map


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--output_dir', type=str, default='render_images')
    parser.add_argument('--edit', type=bool, default=False)
    parser.add_argument('--number', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=9999)
    parser.add_argument('--max_batch_size', type=int, default=2400000)
    parser.add_argument('--lock_view_dependence', action='store_true')
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--output_size', type=int, default=256)
    parser.add_argument('--curriculum', type=str, default='CelebA')
    opt = parser.parse_args()

    curriculum = getattr(curriculums, opt.curriculum)
    curriculum['num_steps'] = curriculum[0]['num_steps']
    curriculum['img_size'] = opt.image_size
    curriculum['output_size'] = opt.output_size
    curriculum['stereo_auxiliary'] = False
    curriculum['psi'] = 0.5
    curriculum['v_stddev'] = 0
    curriculum['h_stddev'] = 0
    curriculum['lock_view_dependence'] = opt.lock_view_dependence
    curriculum['last_back'] = curriculum.get('eval_last_back', False)
    curriculum['nerf_noise'] = 0
    curriculum = {key: value for key, value in curriculum.items() if type(key) is str}

    os.makedirs(opt.output_dir, exist_ok=True)

    SIREN = getattr(siren, curriculum['model'])
    generator = getattr(generators, curriculum['generator'])(SIREN, curriculum['latent_dim'], curriculum['stereo_auxiliary']).to(device)

    ckpt_g = torch.load(opt.path, map_location=torch.device(device))
    generator.load_state_dict(ckpt_g)
    ema1 = ExponentialMovingAverage(generator.decoder.parameters(), decay=0.999)
    ema2 = ExponentialMovingAverage(generator.decoder_parsing.parameters(), decay=0.999)

    ema_file1 = opt.path.replace('generator.pth', 'ema1.pth')
    ema_file2 = opt.path.replace('generator.pth', 'ema2.pth')
    ema1.load_state_dict(torch.load(ema_file1))
    ema2.load_state_dict(torch.load(ema_file2))
    ema1.copy_to(generator.decoder.parameters())
    ema2.copy_to(generator.decoder_parsing.parameters())
    generator.set_device(device)
    generator.eval()

    # face_angles = [0.6, 0.4, 0.2, 0, -0.2, -0.4, -0.6]
    face_angles = []
    start = 0.6
    for i in range(60):
        face_angles.append(start)
        start -= 0.02
    # for t in np.linspace(0, 1, 90):
    #     yaw = 0.4 * np.sin(t * 2 * math.pi)
    #     pitch = 0.2 * np.cos(t * 2 * math.pi)
    #     face_angles.append((pitch, yaw))
        # face_angles.append(yaw)
    face_angles = [a + curriculum['h_mean'] for a in face_angles]

    torch.manual_seed(opt.seed)
    if opt.edit:
        curriculum['h_mean'] = face_angles[5]
        z = torch.randn((1, 256), device=device)
        parsing = getParsing('./editParsing/123456_5_parsing_eye.png')
        parsing = parsing.to(device)
        images, depth_map, parsing_gen, parsing1 = generator.staged_forward_with_parsing(z, 1.0, parsing, **curriculum)
        images = ((images + 1) / 2).float()
        images = images.clamp_(0, 1)
        # print(parsing1)
        parsing = mask2color(parsing)
        parsing1 = mask2color(parsing1)
        save_image(images, os.path.join(opt.output_dir, '{}_edit_image_eye.png'.format(opt.seed)), normalize=True)
        save_image(parsing, os.path.join(opt.output_dir, '{}_edit_parsing_eye.png'.format(opt.seed)), normalize=True)
        save_image(parsing1, os.path.join(opt.output_dir, '{}_edit_parsing1.png'.format(opt.seed)), normalize=True)
    else:
        torch.manual_seed(random.randint(0, opt.seed))
        z1 = torch.randn((1, 256), device=device)
        with torch.no_grad():
            for i, yaw in enumerate(face_angles):

                os.makedirs(os.path.join(opt.output_dir, str(t)), exist_ok=True)

                curriculum['h_mean'] = yaw

                images1, depth1, _, _ = generator.staged_forward(z1, 1.0, **curriculum)

                images1 = ((images1 + 1) / 2).float()
                images1 = images1.clamp_(0, 1)

                save_image(images1, os.path.join(opt.output_dir, f'{i:02d}.png'),
                           normalize=True)
