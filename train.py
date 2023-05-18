import argparse
import os
import math
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from generators import generators
from discriminators import discriminators
from siren import siren
import logging
import datasets
import curriculums
from tqdm import tqdm
import copy
from torch_ema import ExponentialMovingAverage


def mask2color(masks):
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
        18: [0, 204, 0]}

    masks = torch.argmax(masks, dim=1).float()
    sample_mask = torch.zeros((masks.shape[0], masks.shape[1], masks.shape[2], 3), dtype=torch.float)
    for key in COLOR_MAP.keys():
        sample_mask[masks == key] = torch.tensor(COLOR_MAP[key], dtype=torch.float)
    sample_mask = sample_mask.permute(0, 3, 1, 2)
    return sample_mask


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def z_sampler(shape, device, distri):
    z_ = None
    if distri == 'gaussian':
        z_ = torch.randn(shape, device=device)
    elif distri == 'uniform':
        z_ = torch.rand(shape, device=device) * 2 - 1
    return z_


def train(rank, world_size, opt):
    logging.basicConfig(filename=os.path.join(opt.output_dir, 'info.log'), level=logging.INFO)
    torch.manual_seed(0)

    setup(rank, world_size, opt.port)
    device = torch.device(rank)

    logging.info('args = {}'.format(opt))

    curriculum = getattr(curriculums, opt.curriculum)
    metadata = curriculums.extract_metadata(curriculum, 0)

    fix_row = int(4)
    fix_num = int(16)

    fixed_z = z_sampler((fix_num, 256), device='cpu', distri=metadata['z_dist'])

    SIREN = getattr(siren, metadata['model'])

    scaler = torch.cuda.amp.GradScaler()

    if opt.load_dir != '':
        generator = getattr(generators, metadata['generator'])(SIREN, metadata['latent_dim'], metadata['stereo_auxiliary']).to(device)
        discriminator_rgb = getattr(discriminators, metadata['RGB_discriminator'])().to(device)
        discriminator_parsing = getattr(discriminators, metadata['Parsing_discriminator'])(19).to(device)
        ema = ExponentialMovingAverage(generator.parameters(), decay=0.999)
        # please change your file name if you use your own ckpt.
        generator.load_state_dict(torch.load(os.path.join(opt.load_dir, '68000_generator.pth'), map_location=device))
        discriminator_rgb.load_state_dict(torch.load(os.path.join(opt.load_dir, '68000_discriminator_rgb.pth'), map_location=device))
        discriminator_parsing.load_state_dict(torch.load(os.path.join(opt.load_dir, '68000_discriminator_parsing.pth'), map_location=device))
        ema.load_state_dict(torch.load(os.path.join(opt.load_dir, '68000_ema.pth'), map_location=device))
    else:
        generator = getattr(generators, metadata['generator'])(SIREN, metadata['latent_dim'], metadata['stereo_auxiliary']).to(device)
        discriminator_rgb = getattr(discriminators, metadata['RGB_discriminator'])().to(device)
        discriminator_parsing = getattr(discriminators, metadata['Parsing_discriminator'])(19).to(device)
        ema = ExponentialMovingAverage(generator.parameters(), decay=0.999)

    ssim = SSIM().to(device)

    logging.info('generator:')
    logging.info(generator)
    logging.info('discriminator_rgb')
    logging.info(discriminator_rgb)
    logging.info('discriminator_parsing')
    logging.info(discriminator_parsing)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=metadata['gen_lr'], betas=metadata['betas'],
                                   weight_decay=metadata['weight_decay'])
    optimizer_Dr = torch.optim.Adam(discriminator_rgb.parameters(), lr=metadata['disc_lr'], betas=metadata['betas'],
                                    weight_decay=metadata['weight_decay'])
    optimizer_Dp = torch.optim.Adam(discriminator_parsing.parameters(), lr=metadata['disc_lr'], betas=metadata['betas'],
                                    weight_decay=metadata['weight_decay'])

    if opt.load_dir != '':
        optimizer_G.load_state_dict(torch.load(os.path.join(opt.load_dir, '68000_optimizer_G.pth')))
        optimizer_Dr.load_state_dict(torch.load(os.path.join(opt.load_dir, '68000_optimizer_Dr.pth')))
        optimizer_Dp.load_state_dict(torch.load(os.path.join(opt.load_dir, '68000_optimizer_Dp.pth')))
        if not metadata.get('disable_scaler', False):
            scaler.load_state_dict(torch.load(os.path.join(opt.load_dir, '68000_scaler.pth')))

    if opt.set_step is not None:
        generator.step = opt.set_step
        discriminator_rgb.step = opt.set_step
        discriminator_parsing.step = opt.set_step

    if metadata.get('disable_scaler', False):
        scaler = torch.cuda.amp.GradScaler(enabled=False)

    generator.set_device(device)

    # ----------
    #  Training
    # ----------

    torch.manual_seed(rank)
    dataloader = None
    total_progress_bar = tqdm(total=opt.n_epochs, desc="Total progress", dynamic_ncols=True)
    total_progress_bar.update(discriminator_rgb.epoch)
    interior_step_bar = tqdm(dynamic_ncols=True)
    step_last_upsample = None
    for epoch in range(opt.n_epochs):
        total_progress_bar.update(1)
        torch.cuda.empty_cache()
        metadata = curriculums.extract_metadata(curriculum, discriminator_rgb.step)

        for param_group in optimizer_G.param_groups:
            if param_group.get('name', None) == 'mapping_network':
                param_group['lr'] = metadata['gen_lr'] * 5e-2
            else:
                param_group['lr'] = metadata['gen_lr']
            param_group['betas'] = metadata['betas']
            param_group['weight_decay'] = metadata['weight_decay']
        for param_group in optimizer_Dr.param_groups:
            param_group['lr'] = metadata['disc_lr']
            param_group['betas'] = metadata['betas']
            param_group['weight_decay'] = metadata['weight_decay']
        for param_group in optimizer_Dp.param_groups:
            param_group['lr'] = metadata['disc_lr']
            param_group['betas'] = metadata['betas']
            param_group['weight_decay'] = metadata['weight_decay']

        
        if not dataloader or dataloader.batch_size != metadata['batch_size']:
            dataset = datasets.get_dataset(metadata['dataset'], **metadata)
            dataloader = DataLoader(
                dataset,
                batch_size=metadata['batch_size'],
                shuffle=False,
                drop_last=True,
                pin_memory=True,
                num_workers=4,
            )

            step_next_upsample = curriculums.next_upsample_step(curriculum, discriminator_rgb.step)
            step_last_upsample = curriculums.last_upsample_step(curriculum, discriminator_rgb.step)

            interior_step_bar.reset(total=(step_next_upsample - step_last_upsample))
            interior_step_bar.set_description(f"Progress to next stage")
            interior_step_bar.update((discriminator_rgb.step - step_last_upsample))

        # modified

        split_batch_size = metadata['batch_size'] // metadata['batch_split']

        for i, (image, parsing) in enumerate(dataloader):
            metadata = curriculums.extract_metadata(curriculum, discriminator_rgb.step)

            if dataloader.batch_size != metadata['batch_size']:
                break

            if scaler.get_scale() < 1:
                scaler.update(1.)

            generator.train()
            discriminator_rgb.train()
            discriminator_parsing.train()

            alpha = min(1, (discriminator_rgb.step - step_last_upsample) / (metadata['fade_steps']))

            image = image.to(device, non_blocking=True)
            parsing = parsing.to(device, non_blocking=True)

            metadata['nerf_noise'] = max(0, 1. - discriminator_rgb.step / 5000.)

            # TRAIN DISCRIMINATOR_RGB
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    z = z_sampler((image.shape[0], metadata['latent_dim']), device=device, distri=metadata['z_dist'])
                    gen_images = []
                    gen_positions = []

                    for split in range(metadata['batch_split']):
                        subset_z = z[split * split_batch_size:(split + 1) * split_batch_size]
                        g_image, _, g_pos, _, _ = generator(subset_z, alpha, **metadata)

                        gen_images.append(g_image)
                        gen_positions.append(g_pos)

                    gen_images = torch.cat(gen_images, dim=0)
                    gen_positions = torch.cat(gen_positions, dim=0)

                if image.shape != gen_images.shape:
                    dataloader = None
                    break

                image.requires_grad = True
                real_predicts, _ = discriminator_rgb(image, alpha, **metadata)
                gen_predicts, g_pred_position = discriminator_rgb(gen_images, alpha, **metadata)

                if metadata['r1_lambda'] > 0:
                    # Gradient penalty
                    grad_real = torch.autograd.grad(outputs=scaler.scale(real_predicts.sum()), inputs=image,
                                                    create_graph=True)
                    inv_scale = 1. / scaler.get_scale()
                    grad_real = [p * inv_scale for p in grad_real][0]

                    grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
                    grad_penalty_r = 0.5 * metadata['r1_lambda'] * grad_penalty
                else:
                    grad_penalty_r = 0.

                if metadata['pos_lambda'] > 0:
                    identity_penalty_r = torch.nn.MSELoss()(g_pred_position, gen_positions) * metadata['pos_lambda']
                else:
                    identity_penalty_r = 0

                dr_loss = F.softplus(gen_predicts).mean() + F.softplus(-real_predicts).mean() + grad_penalty_r + identity_penalty_r

            optimizer_Dr.zero_grad()
            scaler.scale(dr_loss).backward()
            scaler.unscale_(optimizer_Dr)
            torch.nn.utils.clip_grad_norm_(discriminator_rgb.parameters(), metadata['grad_clip'])
            scaler.step(optimizer_Dr)

            # TRAIN DISCRIMINATOR_Parsing
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    z = z_sampler((image.shape[0], metadata['latent_dim']), device=device, distri=metadata['z_dist'])
                    gen_parsings = []
                    gen_positions = []

                    for split in range(metadata['batch_split']):
                        subset_z = z[split * split_batch_size:(split + 1) * split_batch_size]
                        _, g_parsing, g_pos, _, _ = generator(subset_z, alpha, **metadata)

                        gen_parsings.append(g_parsing)
                        gen_positions.append(g_pos)

                    gen_parsings = torch.cat(gen_parsings, dim=0)
                    gen_positions = torch.cat(gen_positions, dim=0)

                parsing.requires_grad = True
                real_predicts, _ = discriminator_parsing(parsing, alpha, **metadata)
                gen_predicts, g_pred_position = discriminator_parsing(gen_parsings, alpha, **metadata)

                if metadata['r1_lambda'] > 0:
                    # Gradient penalty
                    grad_real = torch.autograd.grad(outputs=scaler.scale(real_predicts.sum()), inputs=parsing,
                                                    create_graph=True)
                    inv_scale = 1. / scaler.get_scale()
                    grad_real = [p * inv_scale for p in grad_real][0]

                    grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
                    grad_penalty_p = 0.5 * metadata['r1_lambda'] * grad_penalty
                else:
                    grad_penalty_p = 0.

                if metadata['pos_lambda'] > 0:
                    identity_penalty_p = torch.nn.MSELoss()(g_pred_position, gen_positions) * metadata['pos_lambda']
                else:
                    identity_penalty_p = 0

                dp_loss = F.softplus(gen_predicts).mean() + F.softplus(-real_predicts).mean() + grad_penalty_p + identity_penalty_p

            optimizer_Dp.zero_grad()
            scaler.scale(dp_loss).backward()
            scaler.unscale_(optimizer_Dp)
            torch.nn.utils.clip_grad_norm_(discriminator_parsing.parameters(), metadata['grad_clip'])
            scaler.step(optimizer_Dp)

            # TRAIN GENERATOR
            z = z_sampler((metadata['batch_size'], metadata['latent_dim']), device=device, distri=metadata['z_dist'])
            topk_percentage = max(0.99 ** (discriminator_rgb.step / metadata['topk_interval']), metadata['topk_v'])

            for split in range(metadata['batch_split']):
                with torch.cuda.amp.autocast():
                    subset_z = z[split * split_batch_size:(split + 1) * split_batch_size]
                    g_image, g_parsing, g_position, g_pri_image, warp_image = generator(subset_z, alpha, **metadata)
                    g_image_predicts, g_image_pred_position = discriminator_rgb(g_image, alpha, **metadata)
                    g_parsing_predicts, g_parsing_pred_position = discriminator_parsing(g_parsing, alpha, **metadata)

                    topk_num = math.ceil(topk_percentage * g_image_predicts.shape[0])

                    g_image_predicts = torch.topk(g_image_predicts, topk_num, dim=0).values
                    g_parsing_predicts = torch.topk(g_parsing_predicts, topk_num, dim=0).values

                    if metadata['pos_lambda'] > 0:
                        identity_penalty_rgb = torch.nn.MSELoss()(g_image_pred_position, g_position) * metadata['pos_lambda']
                        identity_penalty_parsing = torch.nn.MSELoss()(g_parsing_pred_position, g_position) * metadata['pos_lambda']
                        identity_penalty = identity_penalty_rgb + identity_penalty_parsing
                    else:
                        identity_penalty = 0

                    if metadata['reproj_lambda'] > 0:
                        pred = (g_pri_image + 1) / 2
                        target = (warp_image + 1) / 2
                        abs_diff = torch.abs(target - pred)
                        l1_loss = abs_diff.mean(1, True)

                        ssim_loss = ssim(pred, target).mean(1, True)
                        projection_loss = 0.85 * ssim_loss + 0.15 * l1_loss
                        projection_loss = projection_loss.mean() * metadata['reproj_lambda']
                    else:
                        projection_loss = 0

                    g_loss = F.softplus(-g_image_predicts).mean() + identity_penalty + \
                             F.softplus(-g_parsing_predicts).mean() + projection_loss

                scaler.scale(g_loss).backward()

            scaler.unscale_(optimizer_G)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), metadata.get('grad_clip', 0.3))
            scaler.step(optimizer_G)
            scaler.update()
            optimizer_G.zero_grad()
            ema.update(generator.parameters())

            interior_step_bar.update(1)
            if i % 50 == 0:
                tqdm.write(
                    f"[Experiment: {opt.output_dir}] "
                    f"[GPU: {os.environ['CUDA_VISIBLE_DEVICES']}] "
                    f"[Epoch: {discriminator_rgb.epoch}/{opt.n_epochs}] "
                    f"[Dr loss: {dr_loss.item()}] "
                    f"[Dp loss: {dp_loss.item()} "
                    f"[G loss: {g_loss.item()}] "
                    f"[Step: {discriminator_rgb.step}] "
                    f"[Alpha: {alpha:.2f}] "
                    f"[Img Size: {metadata['output_size']}] "
                    f"[Batch Size: {metadata['batch_size']}] "
                    f"[TopK: {topk_num}] "
                    f"[Scale: {scaler.get_scale()}]")
                logging.info(f"[Dr loss: {dr_loss.item()}] "
                             f"[gp_r loss: {grad_penalty_r.item()}] "
                             f"[Dp loss: {dp_loss.item()}] "
                             f"[gp_p loss: {grad_penalty_p.item()}] "
                             f"[G loss: {g_loss.item()}] "
                             f"[project loss: {projection_loss.item()}] "
                             f"[Step: {discriminator_rgb.step}] "
                             f"[Alpha: {alpha:.2f}] "
                             f"[Img Size: {metadata['output_size']}] "
                             f"[Batch Size: {metadata['batch_size']}] "
                             f"[TopK: {topk_num}] "
                             f"[Scale: {scaler.get_scale()}]")
            if discriminator_rgb.step % opt.sample_interval == 0:
                generator.eval()
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        copied_metadata = copy.deepcopy(metadata)
                        copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                        gen_images = []
                        gen_parsings = []
                        for idx in range(fixed_z.shape[0]):
                            g_image, _, g_parsing = generator.staged_forward(fixed_z[idx:idx + 1].to(device), alpha, **copied_metadata)
                            gen_images.append(g_image)
                            gen_parsings.append(g_parsing)
                        gen_images = torch.cat(gen_images, dim=0)
                        gen_parsings = torch.cat(gen_parsings, dim=0)
                        gen_images = ((gen_images + 1) / 2).float()
                        gen_images = gen_images.clamp_(0, 1)
                        gen_parsings = mask2color(gen_parsings)
                save_image(gen_images[:fix_num], os.path.join(opt.output_dir, f"{discriminator_rgb.step}_fixed.png"),
                           nrow=fix_row, normalize=True)
                save_image(gen_parsings[:fix_num], os.path.join(opt.output_dir, f"{discriminator_rgb.step}_fixed_p.png"),
                           nrow=fix_row, normalize=True)
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        copied_metadata = copy.deepcopy(metadata)
                        copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                        copied_metadata['h_mean'] += 0.5
                        gen_images = []
                        gen_parsings = []
                        for idx in range(fixed_z.shape[0]):
                            g_image, _, g_parsing = generator.staged_forward(fixed_z[idx:idx + 1].to(device), alpha, **copied_metadata)
                            gen_images.append(g_image)
                            gen_parsings.append(g_parsing)
                        gen_images = torch.cat(gen_images, dim=0)
                        gen_parsings = torch.cat(gen_parsings, dim=0)
                        gen_images = ((gen_images + 1) / 2).float()
                        gen_images = gen_images.clamp_(0, 1)
                        gen_parsings = mask2color(gen_parsings)
                save_image(gen_images[:fix_num], os.path.join(opt.output_dir, f"{discriminator_rgb.step}_tilted.png"),
                           nrow=fix_row, normalize=True)
                save_image(gen_parsings[:fix_num], os.path.join(opt.output_dir, f"{discriminator_rgb.step}_tilted_p.png"),
                           nrow=fix_row, normalize=True)
                ema.store(generator.parameters())
                ema.copy_to(generator.parameters())
                generator.eval()
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        copied_metadata = copy.deepcopy(metadata)
                        copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                        gen_images = []
                        gen_parsings = []
                        for idx in range(fixed_z.shape[0]):
                            g_image, _, g_parsing = generator.staged_forward(fixed_z[idx:idx + 1].to(device), alpha, **copied_metadata)
                            gen_images.append(g_image)
                            gen_parsings.append(g_parsing)
                        gen_images = torch.cat(gen_images, dim=0)
                        gen_parsings = torch.cat(gen_parsings, dim=0)
                        gen_images = ((gen_images + 1) / 2).float()
                        gen_images = gen_images.clamp_(0, 1)
                        gen_parsings = mask2color(gen_parsings)
                save_image(gen_images[:fix_num], os.path.join(opt.output_dir, f"{discriminator_rgb.step}_fixed_ema.png"),
                           nrow=fix_row, normalize=True)
                save_image(gen_parsings[:fix_num], os.path.join(opt.output_dir, f"{discriminator_rgb.step}_fixed_ema_p.png"),
                           nrow=fix_row, normalize=True)
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        copied_metadata = copy.deepcopy(metadata)
                        copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                        copied_metadata['h_mean'] += 0.5
                        gen_images = []
                        gen_parsings = []
                        for idx in range(fixed_z.shape[0]):
                            g_image, _, g_parsing = generator.staged_forward(fixed_z[idx:idx + 1].to(device), alpha,
                                                                             **copied_metadata)
                            gen_images.append(g_image)
                            gen_parsings.append(g_parsing)
                        gen_images = torch.cat(gen_images, dim=0)
                        gen_parsings = torch.cat(gen_parsings, dim=0)
                        gen_images = ((gen_images + 1) / 2).float()
                        gen_images = gen_images.clamp_(0, 1)
                        gen_parsings = mask2color(gen_parsings)
                save_image(gen_images[:fix_num], os.path.join(opt.output_dir, f"{discriminator_rgb.step}_tilted_ema.png"),
                           nrow=fix_row, normalize=True)
                save_image(gen_parsings[:fix_num], os.path.join(opt.output_dir, f"{discriminator_rgb.step}_tilted_ema_p.png"),
                           nrow=fix_row, normalize=True)

                ema.restore(generator.parameters())

            if discriminator_rgb.step % opt.model_save_interval == 0:
                torch.save(ema.state_dict(), os.path.join(opt.output_dir, '{}_ema.pth'.format(discriminator_rgb.step)))
                torch.save(generator.state_dict(), os.path.join(opt.output_dir, '{}_generator.pth'.format(discriminator_rgb.step)))
                torch.save(discriminator_rgb.state_dict(), os.path.join(opt.output_dir, '{}_discriminator_rgb.pth'.format(discriminator_rgb.step)))
                torch.save(discriminator_parsing.state_dict(), os.path.join(opt.output_dir, '{}_discriminator_parsing.pth'.format(discriminator_rgb.step)))
                torch.save(optimizer_G.state_dict(), os.path.join(opt.output_dir, '{}_optimizer_G.pth'.format(discriminator_rgb.step)))
                torch.save(optimizer_Dr.state_dict(), os.path.join(opt.output_dir, '{}_optimizer_Dr.pth'.format(discriminator_rgb.step)))
                torch.save(optimizer_Dp.state_dict(), os.path.join(opt.output_dir, '{}_optimizer_Dp.pth'.format(discriminator_rgb.step)))
                torch.save(scaler.state_dict(), os.path.join(opt.output_dir, '{}_scaler.pth'.format(discriminator_rgb.step)))

            discriminator_rgb.step += 1
            discriminator_parsing.step += 1
            generator.step += 1
        discriminator_rgb.epoch += 1
        discriminator_parsing.epoch += 1
        generator.epoch += 1

    cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=3000, help="number of epochs of training")
    parser.add_argument("--sample_interval", type=int, default=500, help="interval between image sampling")
    parser.add_argument('--output_dir', type=str, default='debug')
    parser.add_argument('--load_dir', type=str, default='')
    parser.add_argument('--curriculum', type=str, required=True)
    parser.add_argument('--eval_freq', type=int, default=2000)
    parser.add_argument('--port', type=str, default='12355')
    parser.add_argument('--set_step', type=int, default=None)
    parser.add_argument('--model_save_interval', type=int, default=2000)

    option = parser.parse_args()
    print(option)
    os.makedirs(option.output_dir, exist_ok=True)

    train(0, 1, option)
