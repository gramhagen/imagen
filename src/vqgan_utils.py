import math

import kornia.augmentation as K
import numpy as np
from omegaconf import OmegaConf
from PIL import Image, PngImagePlugin
from taming.models import cond_transformer, vqgan
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch_optimizer import DiffGrad, AdamP, RAdam
from torchvision.transforms import functional as TF, Normalize


NORMALIZE_MEAN = [0.48145466, 0.4578275, 0.40821073]
NORMALIZE_STD = [0.26862954, 0.26130258, 0.27577711]
# From imagenet - Which is better?
# NORMALIZE_MEAN = [0.485, 0.456, 0.406]
# NORMALIZE_STD = [0.229, 0.224, 0.225]


normalize = Normalize(NORMALIZE_MEAN, NORMALIZE_STD)


def sinc(x):
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))


def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x / a), x.new_zeros([]))
    return out / out.sum()


def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]


# For zoom video
def zoom_at(img, x, y, zoom):
    w, h = img.size
    zoom2 = zoom * 2
    img = img.crop((x - w / zoom2, y - h / zoom2, x + w / zoom2, y + h / zoom2))
    return img.resize((w, h), Image.LANCZOS)


# NR: Testing with different intital images
def random_noise_image(w, h):
    return Image.fromarray(np.random.randint(0, 255, (w, h, 3), dtype=np.dtype("uint8")))


# create initial gradient image
def gradient_2d(start, stop, width, height, is_horizontal):
    if is_horizontal:
        return np.tile(np.linspace(start, stop, width), (height, 1))
    else:
        return np.tile(np.linspace(start, stop, height), (width, 1)).T


def gradient_3d(width, height, start_list, stop_list, is_horizontal_list):
    result = np.zeros((height, width, len(start_list)), dtype=float)
    for i, (start, stop, is_horizontal) in enumerate(zip(start_list, stop_list, is_horizontal_list)):
        result[:, :, i] = gradient_2d(start, stop, width, height, is_horizontal)
    return result


def random_gradient_image(w, h):
    array = gradient_3d(
        w,
        h,
        (0, 0, np.random.randint(0, 255)),
        (
            np.random.randint(1, 255),
            np.random.randint(2, 255),
            np.random.randint(3, 128),
        ),
        (True, False, False),
    )
    return Image.fromarray(np.uint8(array))


# Used in older MakeCutouts
def resample(input, size, align_corners=True):
    n, c, h, w = input.shape
    dh, dw = size

    input = input.view([n * c, 1, h, w])

    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), "reflect")
        input = F.conv2d(input, kernel_h[None, None, :, None])

    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), "reflect")
        input = F.conv2d(input, kernel_w[None, None, None, :])

    input = input.view([n, c, h, w])
    return F.interpolate(input, size, mode="bicubic", align_corners=align_corners)


class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)


class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_in):
        (input,) = ctx.saved_tensors
        return (
            grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0),
            None,
            None,
        )


def vector_quantize(x, codebook):
    d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
    indices = d.argmin(-1)
    x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
    return ReplaceGrad.apply(x_q, x)


class Prompt(nn.Module):
    def __init__(self, embed, weight=1.0, stop=float("-inf")):
        super().__init__()
        self.register_buffer("embed", embed)
        self.register_buffer("weight", torch.as_tensor(weight))
        self.register_buffer("stop", torch.as_tensor(stop))

    def forward(self, input):
        input_normed = F.normalize(input.unsqueeze(1), dim=2)
        embed_normed = F.normalize(self.embed.unsqueeze(0), dim=2)
        dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        dists = dists * self.weight.sign()
        return self.weight.abs() * ReplaceGrad.apply(dists, torch.maximum(dists, self.stop)).mean()


# NR: Split prompts and weights
def split_prompt(prompt):
    vals = prompt.rsplit(":", 2)
    vals = vals + ["", "1", "-inf"][len(vals) :]
    return vals[0], float(vals[1]), float(vals[2])


# This is the original version (No pooling)
class MakeCutoutsOrig(nn.Module):
    def __init__(self, augments, cut_size, cutn, cut_pow=1.0):
        super().__init__()
        self.augments = augments
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow  # not used with pooling

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([]) ** self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety : offsety + size, offsetx : offsetx + size]
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
        return ClampWithGrad.apply(torch.cat(cutouts, dim=0), 0, 1)


class MakeCutouts(MakeCutoutsOrig):
    def __init__(self, augments, cut_size, cutn, cut_pow=1.0):
        super().__init__(augments, cut_size, cutn, cut_pow)

        # Pick your own augments & their order
        augment_list = []
        for item in self.augments[0]:
            if item == "Ji":
                augment_list.append(K.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.7))
            elif item == "Sh":
                augment_list.append(K.RandomSharpness(sharpness=0.3, p=0.5))
            elif item == "Gn":
                augment_list.append(K.RandomGaussianNoise(mean=0.0, std=1.0, p=0.5))
            elif item == "Pe":
                augment_list.append(K.RandomPerspective(distortion_scale=0.7, p=0.7))
            elif item == "Ro":
                augment_list.append(K.RandomRotation(degrees=15, p=0.7))
            elif item == "Af":
                augment_list.append(
                    K.RandomAffine(
                        degrees=15,
                        translate=0.1,
                        shear=5,
                        p=0.7,
                        padding_mode="zeros",
                        keepdim=True,
                    )
                )  # border, reflection, zeros
            elif item == "Et":
                augment_list.append(K.RandomElasticTransform(p=0.7))
            elif item == "Ts":
                augment_list.append(K.RandomThinPlateSpline(scale=0.8, same_on_batch=True, p=0.7))
            elif item == "Cr":
                augment_list.append(
                    K.RandomCrop(
                        size=(self.cut_size, self.cut_size),
                        pad_if_needed=True,
                        padding_mode="reflect",
                        p=0.5,
                    )
                )
            elif item == "Er":
                augment_list.append(
                    K.RandomErasing(
                        scale=(0.1, 0.4),
                        ratio=(0.3, 1 / 0.3),
                        same_on_batch=True,
                        p=0.7,
                    )
                )
            elif item == "Re":
                augment_list.append(
                    K.RandomResizedCrop(
                        size=(self.cut_size, self.cut_size),
                        scale=(0.1, 1),
                        ratio=(0.75, 1.333),
                        cropping_mode="resample",
                        p=0.5,
                    )
                )

        self.augs = nn.Sequential(*augment_list)
        self.noise_fac = 0.1
        # self.noise_fac = False

        # Uncomment if you like seeing the list ;)
        # print(augment_list)

        # Pooling
        self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
        self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))

    def forward(self, input):
        cutouts = []

        for _ in range(self.cutn):
            # Use Pooling
            cutout = (self.av_pool(input) + self.max_pool(input)) / 2
            cutouts.append(cutout)

        batch = self.augs(torch.cat(cutouts, dim=0))

        if self.noise_fac:
            facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch


# An updated version with Kornia augments and pooling (where my version started):
class MakeCutoutsPoolingUpdate(MakeCutoutsOrig):
    def __init__(self, augments, cut_size, cutn, cut_pow=1.0):
        super().__init__(augments, cut_size, cutn, cut_pow)

        self.augs = nn.Sequential(
            K.RandomAffine(degrees=15, translate=0.1, p=0.7, padding_mode="border"),
            K.RandomPerspective(0.7, p=0.7),
            K.ColorJitter(hue=0.1, saturation=0.1, p=0.7),
            K.RandomErasing((0.1, 0.4), (0.3, 1 / 0.3), same_on_batch=True, p=0.7),
        )

        self.noise_fac = 0.1
        self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
        self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []

        for _ in range(self.cutn):
            cutout = (self.av_pool(input) + self.max_pool(input)) / 2
            cutouts.append(cutout)

        batch = self.augs(torch.cat(cutouts, dim=0))

        if self.noise_fac:
            facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch


# An Nerdy updated version with selectable Kornia augments, but no pooling:
class MakeCutoutsNRUpdate(MakeCutoutsOrig):
    def __init__(self, augments, cut_size, cutn, cut_pow=1.0):
        super().__init__(augments, cut_size, cutn, cut_pow)
        self.noise_fac = 0.1

        # Pick your own augments & their order
        augment_list = []
        for item in self.augments[0]:
            if item == "Ji":
                augment_list.append(K.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.7))
            elif item == "Sh":
                augment_list.append(K.RandomSharpness(sharpness=0.3, p=0.5))
            elif item == "Gn":
                augment_list.append(K.RandomGaussianNoise(mean=0.0, std=1.0, p=0.5))
            elif item == "Pe":
                augment_list.append(K.RandomPerspective(distortion_scale=0.5, p=0.7))
            elif item == "Ro":
                augment_list.append(K.RandomRotation(degrees=15, p=0.7))
            elif item == "Af":
                augment_list.append(
                    K.RandomAffine(
                        degrees=30,
                        translate=0.1,
                        shear=5,
                        p=0.7,
                        padding_mode="zeros",
                        keepdim=True,
                    )
                )  # border, reflection, zeros
            elif item == "Et":
                augment_list.append(K.RandomElasticTransform(p=0.7))
            elif item == "Ts":
                augment_list.append(K.RandomThinPlateSpline(scale=0.8, same_on_batch=True, p=0.7))
            elif item == "Cr":
                augment_list.append(
                    K.RandomCrop(
                        size=(self.cut_size, self.cut_size),
                        pad_if_needed=True,
                        padding_mode="reflect",
                        p=0.5,
                    )
                )
            elif item == "Er":
                augment_list.append(
                    K.RandomErasing(
                        scale=(0.1, 0.4),
                        ratio=(0.3, 1 / 0.3),
                        same_on_batch=True,
                        p=0.7,
                    )
                )
            elif item == "Re":
                augment_list.append(
                    K.RandomResizedCrop(
                        size=(self.cut_size, self.cut_size),
                        scale=(0.1, 1),
                        ratio=(0.75, 1.333),
                        cropping_mode="resample",
                        p=0.5,
                    )
                )

        self.augs = nn.Sequential(*augment_list)

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([]) ** self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety : offsety + size, offsetx : offsetx + size]
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
        batch = self.augs(torch.cat(cutouts, dim=0))
        if self.noise_fac:
            facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch


# An updated version with Kornia augments, but no pooling:
class MakeCutoutsUpdate(MakeCutoutsOrig):
    def __init__(self, augments, cut_size, cutn, cut_pow=1.0):
        super().__init__(augments, cut_size, cutn, cut_pow)
        self.augs = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            K.ColorJitter(hue=0.01, saturation=0.01, p=0.7),
            # K.RandomSolarize(0.01, 0.01, p=0.7),
            K.RandomSharpness(0.3, p=0.4),
            K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode="border"),
            K.RandomPerspective(0.2, p=0.4),
        )
        self.noise_fac = 0.1

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([]) ** self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety : offsety + size, offsetx : offsetx + size]
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
        batch = self.augs(torch.cat(cutouts, dim=0))
        if self.noise_fac:
            facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch


def load_vqgan_model(device, config_path):
    gumbel = False
    checkpoint_path = config_path.replace(".yaml", ".ckpt")
    config = OmegaConf.load(config_path)
    if config.model.target == "taming.models.vqgan.VQModel":
        model = vqgan.VQModel(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == "taming.models.vqgan.GumbelVQ":
        model = vqgan.GumbelVQ(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
        gumbel = True
    elif config.model.target == "taming.models.cond_transformer.Net2NetTransformer":
        parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
        parent_model.eval().requires_grad_(False)
        parent_model.init_from_ckpt(checkpoint_path)
        model = parent_model.first_stage_model
    else:
        raise ValueError(f"unknown model type: {config.model.target}")
    del model.loss
    return (model.to(device), gumbel)


def resize_image(image, out_size):
    ratio = image.size[0] / image.size[1]
    area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
    size = round((area * ratio) ** 0.5), round((area / ratio) ** 0.5)
    return image.resize(size, Image.LANCZOS)


# Set the optimiser
def get_opt(opt_name, z, opt_lr):
    if opt_name == "Adam":
        opt = optim.Adam([z], lr=opt_lr)  # LR=0.1 (Default)
    elif opt_name == "AdamW":
        opt = optim.AdamW([z], lr=opt_lr)
    elif opt_name == "Adagrad":
        opt = optim.Adagrad([z], lr=opt_lr)
    elif opt_name == "Adamax":
        opt = optim.Adamax([z], lr=opt_lr)
    elif opt_name == "DiffGrad":
        opt = DiffGrad([z], lr=opt_lr, eps=1e-9, weight_decay=1e-9)  # NR: Playing for reasons
    elif opt_name == "AdamP":
        opt = AdamP([z], lr=opt_lr)
    elif opt_name == "RAdam":
        opt = RAdam([z], lr=opt_lr)
    elif opt_name == "RMSprop":
        opt = optim.RMSprop([z], lr=opt_lr)
    else:
        print("Unknown optimiser. Are choices broken?")
        opt = optim.Adam([z], lr=opt_lr)
    return opt


# Vector quantize
def synth(z, gumbel, model):
    if gumbel:
        z_q = vector_quantize(z.movedim(1, 3), model.quantize.embed.weight).movedim(3, 1)
    else:
        z_q = vector_quantize(z.movedim(1, 3), model.quantize.embedding.weight).movedim(3, 1)
    return ClampWithGrad.apply(model.decode(z_q).add(1).div(2), 0, 1)


# @torch.no_grad()
@torch.inference_mode()
def checkin(i, losses, out, args):
    losses_str = ", ".join(f"{loss.item():g}" for loss in losses)
    tqdm.write(f"i: {i}, loss: {sum(losses).item():g}, losses: {losses_str}")
    info = PngImagePlugin.PngInfo()
    info.add_text("comment", f"{args.prompts}")
    TF.to_pil_image(out[0].cpu()).save(args.output, pnginfo=info)


def ascend_txt(i, pMs, perceptor, make_cutouts, out, z, z_orig, args):
    encoding = perceptor.encode_image(normalize(make_cutouts(out))).float()

    result = []
    if args.init_weight:
        # result.append(F.mse_loss(z, z_orig) * init_weight / 2)
        result.append(F.mse_loss(z, torch.zeros_like(z_orig)) * ((1 / torch.tensor(i * 2 + 1)) * args.init_weight) / 2)

    for prompt in pMs:
        result.append(prompt(encoding))

    if args.make_video:
        img = np.array(out.mul(255).clamp(0, 255)[0].cpu().detach().numpy().astype(np.uint8))[:, :, :]
        img = np.transpose(img, (1, 2, 0))
        imageio.imwrite("./steps/" + str(i) + ".png", np.array(img))

    return result  # return loss


def train(i, pMs, opt, perceptor, z, z_orig, z_min, z_max, gumbel, model, make_cutouts, args):
    opt.zero_grad(set_to_none=True)
    out = synth(z, gumbel, model)
    lossAll = ascend_txt(i, pMs, perceptor, make_cutouts, out, z, z_orig, args)

    if i % args.display_freq == 0:
        checkin(i, lossAll, out, args)

    loss = sum(lossAll)
    loss.backward()
    opt.step()

    # with torch.no_grad():
    with torch.inference_mode():
        z.copy_(z.maximum(z_min).minimum(z_max))
