from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import ListConfig
from torchvision.utils import save_image
from ...util import append_dims, instantiate_from_config
import torchvision.models as models


def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a* b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

class ResNetFeatures(nn.Module):
    def __init__(self):
        super(ResNetFeatures, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-2])  
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        return x

# class VGGFeatures(nn.Module):
#     def __init__(self):
#         super(VGGFeatures, self).__init__()
#         vgg = vgg19(pretrained=True).features
#         self.layers = {'0': 'relu1_1', '5': 'relu2_1', '10': 'relu3_1', '19': 'relu4_1'}
#         self.features = nn.ModuleDict({
#             name: vgg[:int(layer) + 1] for layer, name in self.layers.items()
#         })
#         for param in self.parameters():
#             param.requires_grad = False

    # def forward(self, x):
    #     outputs = {}
    #     for name, model in self.features.items():
    #         x = model(x)
    #         outputs[name] = x
    #     return outputs


class StandardDiffusionLoss(nn.Module):
    def __init__(
        self,
        sigma_sampler_config,
        type="huber",
        offset_noise_level=0.0,
        batch2model_keys: Optional[Union[str, List[str], ListConfig]] = None,
    ):
        super().__init__()

        assert type in ["l2", "l1", "huber"]

        self.sigma_sampler = instantiate_from_config(sigma_sampler_config)

        self.type = type
        self.offset_noise_level = offset_noise_level

        if not batch2model_keys:
            batch2model_keys = []

        if isinstance(batch2model_keys, str):
            batch2model_keys = [batch2model_keys]

        self.batch2model_keys = set(batch2model_keys)

    def __call__(self, network, denoiser, conditioner, input, batch, *args, **kwarg):
        # print("Batch contents:", batch)
        cond = conditioner(batch)
        additional_model_inputs = {
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }
        # print(additional_model_inputs)
        sigmas = self.sigma_sampler(input.shape[0]).to(input.device)
        noise = torch.randn_like(input)
        if self.offset_noise_level > 0.0:
            noise = noise + self.offset_noise_level * append_dims(
                torch.randn(input.shape[0], device=input.device), input.ndim
            )
        noised_input = input + noise * append_dims(sigmas, input.ndim)
        model_output = denoiser(
            network, noised_input, sigmas, cond, **additional_model_inputs
        )
        w = append_dims(denoiser.w(sigmas), input.ndim)

        loss = self.get_diff_loss(model_output, input, w)
        loss = loss.mean()
        loss_dict = {"loss": loss}

        return loss, loss_dict

    def get_diff_loss(self, model_output, target, w):
        if self.type == "l2":
            return torch.mean(
                (w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1
            )
        elif self.type == "l1":
            return torch.mean(
                (w * (model_output - target).abs()).reshape(target.shape[0], -1), 1
            )
        elif self.type == "huber":
            print("Using Huber loss in diff loss")
            diff = torch.abs(model_output - target)
            loss = torch.where(diff < 1, 0.5 * (diff ** 2) / 1, diff - 0.5 * 1)
            return torch.mean(w * loss).reshape(target.shape[0], -1)
            
        # elif self.type == "huber":
        #     return torch.mean(
        #         (w * F.huber_loss(model_output, target)).reshape(target.shape[0], -1), 1
        #     )


class FullLoss(StandardDiffusionLoss):

    def __init__(
        self,
        seq_len=12,
        kernel_size=3,
        gaussian_sigma=0.5,
        min_attn_size=16,
        lambda_local_loss=0.0,
        lambda_ocr_loss=0.0,
        lambda_style_loss=0.0,
        lambda_mask_loss=0.0,
        ocr_enabled = False,
        style_enabled = True,
        mask_enabled = False,
        predictor_config = None,
        input_key: str = "image",
        *args, **kwarg
    ):
        super().__init__(*args, **kwarg)
        self.input_key = input_key
        self.resnet = ResNetFeatures().cuda()
        self.gaussian_kernel_size = kernel_size
        gaussian_kernel = self.get_gaussian_kernel(kernel_size=self.gaussian_kernel_size, sigma=gaussian_sigma, out_channels=seq_len)
        self.register_buffer("g_kernel", gaussian_kernel.requires_grad_(False))

        self.min_attn_size = min_attn_size
        self.lambda_local_loss = lambda_local_loss
        self.lambda_ocr_loss = lambda_ocr_loss
        self.lambda_style_loss = lambda_style_loss
        self.lambda_mask_loss = lambda_mask_loss
        self.mask_enabled= mask_enabled
        self.style_enabled = style_enabled
        self.ocr_enabled = ocr_enabled
        if ocr_enabled:
            self.predictor = instantiate_from_config(predictor_config)
    
    def get_gaussian_kernel(self, kernel_size=3, sigma=1, out_channels=3):
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (kernel_size - 1)/2.
        variance = sigma**2.

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1./(2.*torch.pi*variance)) *\
                        torch.exp(
                            -torch.sum((xy_grid - mean)**2., dim=-1) /\
                            (2*variance)
                        )

        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.tile(out_channels, 1, 1, 1)
        
        return gaussian_kernel

    def __call__(self, network, denoiser, conditioner, input, batch, first_stage_model, scaler):

        # print("Batch contents:", batch)
        cond = conditioner(batch)

        sigmas = self.sigma_sampler(input.shape[0]).to(input.device)
        noise = torch.randn_like(input)
        if self.offset_noise_level > 0.0:
            noise = noise + self.offset_noise_level * append_dims(
                torch.randn(input.shape[0], device=input.device), input.ndim
            )
        noised_input = input + noise * append_dims(sigmas, input.ndim)
        model_output = denoiser(network, noised_input, sigmas, cond)
        w = append_dims(denoiser.w(sigmas), input.ndim)

        diff_loss = self.get_diff_loss(model_output, input, w)
        local_loss = self.get_local_loss(network.diffusion_model.attn_map_cache, batch["seg"], batch["seg_mask"])
        diff_loss = diff_loss.mean()
        local_loss = local_loss.mean()

        if self.mask_enabled:
            mask_loss = self.get_mask_local_loss(network.diffusion_model.attn_map_cache, batch["mask"])
            # print("Mask enabled")
        
        if self.ocr_enabled:
            ocr_loss = self.get_ocr_loss(model_output, batch["r_bbox"], batch["label"], first_stage_model, scaler)
            ocr_loss = ocr_loss.mean()

        if self.style_enabled:

            original_crops = self.get_cropped_images(batch[self.input_key], batch["r_bbox"])
            print("Getting model output")
            model_output = 1 / scaler * model_output
            model_output_decoded = first_stage_model.decode(model_output)
            gen_crops = self.get_cropped_images(model_output_decoded, batch["r_bbox"])
            
            total_style_loss = self.get_style_local_loss(gen_crops, original_crops)
            

        loss = diff_loss + self.lambda_local_loss * local_loss
        
        if self.ocr_enabled:
            loss += self.lambda_ocr_loss * ocr_loss
        if self.style_enabled:
            loss += self.lambda_style_loss * total_style_loss
        if self.mask_enabled:
            loss += self.lambda_mask_loss * mask_loss

        loss_dict = {
            "loss/diff_loss": diff_loss,
            "loss/local_loss": local_loss,
            "loss/full_loss": loss
        }

        if self.ocr_enabled:
            loss_dict["loss/ocr_loss"] = ocr_loss
        if self.style_enabled:
            loss_dict["loss/style_loss"] = total_style_loss
        if self.mask_enabled:
            loss_dict["loss/mask_loss"] = mask_loss

        return loss, loss_dict

    def get_cropped_images(self, images, bboxes):
        crops = []
        for i, bbox in enumerate(bboxes):
            m_top, m_bottom, m_left, m_right = bbox
            crop = images[:, :, m_top:m_bottom, m_left:m_right]
            crop_features = self.resnet(crop)
            crops.append(crop_features)
        return crops
        

    def get_mask_local_loss(self, attn_map_cache, mask):
        mask_loss = 0
        count = 0
    
        for item in attn_map_cache:
            if not item["name"].endswith("t_attn"):
                continue
    
            attn_map = item["attn_map"]
            if attn_map is None:
                continue
    
            resized_mask = F.interpolate(mask, size=attn_map.shape[-2:], mode='nearest')
            mask_loss += (attn_map * resized_mask).mean()
            count += 1
    
        if count > 0:
            mask_loss /= count

        return mask_loss
    
    def get_style_local_loss(self,gen_crop, original_crop):

        style_loss = 0
        num=0
        for gen_f, orig_f in zip(gen_crop, original_crop):
            gm_gen = gram_matrix(gen_f)
            gm_orig = gram_matrix(orig_f)
            style_loss += F.mse_loss(gm_gen, gm_orig)
            num+=1

        return style_loss/num
    
    def get_ocr_loss(self, model_output, r_bbox, label, first_stage_model, scaler):

        model_output = 1 / scaler * model_output
        model_output_decoded = first_stage_model.decode(model_output)
        model_output_crops = []
        
        for i, bbox in enumerate(r_bbox):
            m_top, m_bottom, m_left, m_right = bbox
            model_output_crops.append(model_output_decoded[i, :, m_top:m_bottom, m_left:m_right])

        loss = self.predictor.calc_loss(model_output_crops, label)

        return loss

    def get_min_local_loss(self, attn_map_cache, mask, seg_mask):

        loss = 0
        count = 0

        for item in attn_map_cache:

            name = item["name"]
            if not name.endswith("t_attn"): continue

            heads = item["heads"]
            size = item["size"]
            attn_map = item["attn_map"]

            if size < self.min_attn_size: continue

            seg_l = seg_mask.shape[1]

            bh, n, l = attn_map.shape # bh: batch size * heads / n : pixel length(h*w) / l: token length
            attn_map = attn_map.reshape((-1, heads, n, l)) # b, h, n, l
            
            assert seg_l <= l
            attn_map = attn_map[..., :seg_l]
            attn_map = attn_map.permute(0, 1, 3, 2) # b, h, l, n
            attn_map = attn_map.mean(dim = 1) # b, l, n

            attn_map = attn_map.reshape((-1, seg_l, size, size)) # b, l, s, s
            attn_map = F.conv2d(attn_map, self.g_kernel, padding = self.gaussian_kernel_size//2, groups=seg_l) # gaussian blur on each channel
            attn_map = attn_map.reshape((-1, seg_l, n)) # b, l, n
            
            mask_map = F.interpolate(mask, (size, size))
            mask_map = mask_map.tile((1, seg_l, 1, 1))
            mask_map = mask_map.reshape((-1, seg_l, n)) # b, l, n

            p_loss = (mask_map * attn_map).max(dim = -1)[0] # b, l
            p_loss = p_loss + (1 - seg_mask) # b, l
            p_loss = p_loss.min(dim = -1)[0] # b,

            loss += -p_loss
            count += 1

        loss = loss / count

        return loss

    def get_local_loss(self, attn_map_cache, seg, seg_mask):

        loss = 0
        count = 0

        for item in attn_map_cache:

            name = item["name"]
            if not name.endswith("t_attn"): continue

            heads = item["heads"]
            size = item["size"]
            attn_map = item["attn_map"]

            if size < self.min_attn_size: continue

            seg_l = seg_mask.shape[1]

            bh, n, l = attn_map.shape # bh: batch size * heads / n: pixel length(h*w) / l: token length
            attn_map = attn_map.reshape((-1, heads, n, l)) # b, h, n, l
            
            assert seg_l <= l
            attn_map = attn_map[..., :seg_l]
            attn_map = attn_map.permute(0, 1, 3, 2) # b, h, l, n
            attn_map = attn_map.mean(dim = 1) # b, l, n

            attn_map = attn_map.reshape((-1, seg_l, size, size)) # b, l, s, s
            attn_map = F.conv2d(attn_map, self.g_kernel, padding = self.gaussian_kernel_size//2, groups=seg_l) # gaussian blur on each channel
            attn_map = attn_map.reshape((-1, seg_l, n)) # b, l, n

            seg_map = F.interpolate(seg, (size, size))
            seg_map = seg_map.reshape((-1, seg_l, n)) # b, l, n
            n_seg_map = 1 - seg_map

            p_loss = (seg_map * attn_map).max(dim = -1)[0] # b, l
            n_loss = (n_seg_map * attn_map).max(dim = -1)[0] # b, l

            p_loss = p_loss * seg_mask # b, l
            n_loss = n_loss * seg_mask # b, l

            p_loss = p_loss.sum(dim = -1) / seg_mask.sum(dim = -1) # b,
            n_loss = n_loss.sum(dim = -1) / seg_mask.sum(dim = -1) # b,

            f_loss = n_loss - p_loss # b,
            loss += f_loss
            count += 1

        loss = loss / count

        return loss
    
