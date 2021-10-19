# Refactored from https://github.com/nerdyrodent/VQGAN-CLIP/blob/main/generate.py
# Originally made by Katherine Crowson (https://github.com/crowsonkb, https://twitter.com/RiversHaveWings)
# The original BigGAN+CLIP method was by https://twitter.com/advadnoun

import random
from urllib.request import urlopen
from tqdm import tqdm
import os
import re
from subprocess import Popen, PIPE
import warnings

# Supress warnings
warnings.filterwarnings("ignore")

import clip
import imageio
import numpy as np
from PIL import ImageFile, Image, ImageChops
from pydantic import BaseModel
import torch
from torch import optim
from torch.nn import functional as F
from torchvision.transforms import functional as TF

from vqgan_utils import *


torch.backends.cudnn.benchmark = False  # NR: True is a bit faster, but can lead to OOM. False is more deterministic.
# torch.use_deterministic_algorithms(True)	# NR: grid_sampler_2d_backward_cuda does not have a deterministic implementation

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Check for GPU and reduce the default image size if low VRAM
IMAGE_SIZE = 512  # >8GB VRAM
if not torch.cuda.is_available():
    IMAGE_SIZE = 256  # no GPU found
elif torch.cuda.get_device_properties(0).total_memory <= 2 ** 33:  # 2 ** 33 = 8,589,934,592 bytes = 8 GB
    IMAGE_SIZE = 318  # <8GB VRAM


class VQArgs(BaseModel):
    prompts: str = None
    image_prompts: list = []
    max_iterations: int = 500
    display_freq: int = 50
    size: int = IMAGE_SIZE
    init_image: str = None
    init_noise: str = None
    init_weight: float = 0.0
    clip_model: str = "ViT-B/32"
    vqgan_config: str = "checkpoints/vqgan_imagenet_f16_16384.yaml"
    noise_prompt_seeds: list = []
    noise_prompt_weights: list = []
    step_size: float = 0.1
    cut_method: str = "latest"
    cutn: int = 32
    cut_pow: float = 1.0
    seed: int = None
    optimiser: str = "Adam"
    output: str = "output/image.png"
    make_video: bool = False
    make_zoom_video: bool = False
    zoom_start: int = 0
    zoom_frequency: int = 10
    zoom_scale: float = 0.99
    zoom_shift_x: int = 0
    zoom_shift_y: int = 0
    prompt_frequency: int = 0
    video_length: float = 10.0
    output_video_fps: float = 0.0
    input_video_fps: float = 15.0
    cudnn_determinism: bool = False
    augments: list = []
    video_style_dir: str = None
    cuda_device: str = "cuda:0"


def generate(args: VQArgs):
    # check for lock file


    if not args.prompts and not args.image_prompts:
        args.prompts = "psychedelic space chickens"

    if args.cudnn_determinism:
        torch.backends.cudnn.deterministic = True

    if not args.augments:
        args.augments = [["Af", "Pe", "Ji", "Er"]]

    # Split text prompts using the pipe character (weights are split later)
    if args.prompts:
        # For stories, there will be many phrases
        story_phrases = [phrase.strip() for phrase in args.prompts.split("^")]

        # Make a list of all phrases
        all_phrases = []
        for phrase in story_phrases:
            all_phrases.append(phrase.split("|"))

        # First phrase
        args.prompts = all_phrases[0]

    # Split target images using the pipe character (weights are split later)
    if args.image_prompts:
        args.image_prompts = args.image_prompts.split("|")
        args.image_prompts = [image.strip() for image in args.image_prompts]

    if args.make_video and args.make_zoom_video:
        print("Warning: Make video and make zoom video are mutually exclusive.")
        args.make_video = False

    # Make video steps directory
    if args.make_video or args.make_zoom_video:
        if not os.path.exists("steps"):
            os.mkdir("steps")

    # Fallback to CPU if CUDA is not found and make sure GPU video rendering is also disabled
    # NB. May not work for AMD cards?
    if not args.cuda_device == "cpu" and not torch.cuda.is_available():
        args.cuda_device = "cpu"
        args.output_video_fps = 0
        print("Warning: No GPU found! Using the CPU instead. The iterations will be slow.")
        print("Perhaps CUDA/ROCm or the right pytorch version is not properly installed?")

    # If a video_style_dir has been, then create a list of all the images
    if args.video_style_dir:
        print("Locating video frames...")
        video_frame_list = []
        for entry in os.scandir(args.video_style_dir):
            if (entry.path.endswith(".jpg") or entry.path.endswith(".png")) and entry.is_file():
                video_frame_list.append(entry.path)

        # Reset a few options - same filename, different directory
        if not os.path.exists("steps"):
            os.mkdir("steps")

        args.init_image = video_frame_list[0]
        filename = os.path.basename(args.init_image)
        cwd = os.getcwd()
        args.output = os.path.join(cwd, "steps", filename)
        num_video_frames = len(video_frame_list)  # for video styling

    # Load models
    device = torch.device(args.cuda_device)
    model, gumbel = load_vqgan_model(device, args.vqgan_config)
    jit = True if float(torch.__version__[:3]) < 1.8 else False
    perceptor = clip.load(args.clip_model, jit=jit)[0].eval().requires_grad_(False).to(device)

    # clock=deepcopy(perceptor.visual.positional_embedding.data)
    # perceptor.visual.positional_embedding.data = clock/clock.max()
    # perceptor.visual.positional_embedding.data=clamp_with_grad(clock,0,1)

    cut_size = perceptor.visual.input_resolution
    f = 2 ** (model.decoder.num_resolutions - 1)

    # Cutout class options:
    # 'latest','original','updated' or 'updatedpooling'
    if args.cut_method == "latest":
        make_cutouts = MakeCutouts(args.augments, cut_size, args.cutn, cut_pow=args.cut_pow)
    elif args.cut_method == "original":
        make_cutouts = MakeCutoutsOrig(args.augments, cut_size, args.cutn, cut_pow=args.cut_pow)
    elif args.cut_method == "updated":
        make_cutouts = MakeCutoutsUpdate(args.augments, cut_size, args.cutn, cut_pow=args.cut_pow)
    elif args.cut_method == "nrupdated":
        make_cutouts = MakeCutoutsNRUpdate(args.augments, cut_size, args.cutn, cut_pow=args.cut_pow)
    else:
        make_cutouts = MakeCutoutsPoolingUpdate(args.augments, cut_size, args.cutn, cut_pow=args.cut_pow)

    image_size = [args.size, args.size]
    toksX, toksY = image_size[0] // f, image_size[1] // f
    sideX, sideY = toksX * f, toksY * f

    # Gumbel or not?
    if gumbel:
        e_dim = 256
        n_toks = model.quantize.n_embed
        z_min = model.quantize.embed.weight.min(dim=0).values[None, :, None, None]
        z_max = model.quantize.embed.weight.max(dim=0).values[None, :, None, None]
    else:
        e_dim = model.quantize.e_dim
        n_toks = model.quantize.n_e
        z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
        z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]

    if args.init_image:
        if "http" in args.init_image:
            img = Image.open(urlopen(args.init_image))
        else:
            img = Image.open(args.init_image)
            pil_image = img.convert("RGB")
            pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
            pil_tensor = TF.to_tensor(pil_image)
            z, *_ = model.encode(pil_tensor.to(device).unsqueeze(0) * 2 - 1)
    elif args.init_noise == "pixels":
        img = random_noise_image(image_size[0], image_size[1])
        pil_image = img.convert("RGB")
        pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
        pil_tensor = TF.to_tensor(pil_image)
        z, *_ = model.encode(pil_tensor.to(device).unsqueeze(0) * 2 - 1)
    elif args.init_noise == "gradient":
        img = random_gradient_image(image_size[0], image_size[1])
        pil_image = img.convert("RGB")
        pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
        pil_tensor = TF.to_tensor(pil_image)
        z, *_ = model.encode(pil_tensor.to(device).unsqueeze(0) * 2 - 1)
    else:
        one_hot = F.one_hot(torch.randint(n_toks, [toksY * toksX], device=device), n_toks).float()
        # z = one_hot @ model.quantize.embedding.weight
        if gumbel:
            z = one_hot @ model.quantize.embed.weight
        else:
            z = one_hot @ model.quantize.embedding.weight

        z = z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)
        # z = torch.rand_like(z)*2						# NR: check

    z_orig = z.clone()
    z.requires_grad_(True)

    pMs = []

    # CLIP tokenize/encode
    if args.prompts:
        for prompt in args.prompts:
            txt, weight, stop = split_prompt(prompt)
            embed = perceptor.encode_text(clip.tokenize(txt).to(device)).float()
            pMs.append(Prompt(embed, weight, stop).to(device))

    for prompt in args.image_prompts:
        path, weight, stop = split_prompt(prompt)
        img = Image.open(path)
        pil_image = img.convert("RGB")
        img = resize_image(pil_image, (sideX, sideY))
        batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(device))
        embed = perceptor.encode_image(normalize(batch)).float()
        pMs.append(Prompt(embed, weight, stop).to(device))

    for seed, weight in zip(args.noise_prompt_seeds, args.noise_prompt_weights):
        gen = torch.Generator().manual_seed(seed)
        embed = torch.empty([1, perceptor.visual.output_dim]).normal_(generator=gen)
        pMs.append(Prompt(embed, weight).to(device))

    opt = get_opt(args.optimiser, z, args.step_size)

    # Output for the user
    print("Using device:", device)
    print("Optimising using:", args.optimiser)

    if args.prompts:
        print("Using text prompts:", args.prompts)
    if args.image_prompts:
        print("Using image prompts:", args.image_prompts)
    if args.init_image:
        print("Using initial image:", args.init_image)
    if args.noise_prompt_weights:
        print("Noise prompt weights:", args.noise_prompt_weights)

    if args.seed is None:
        seed = torch.seed()
    else:
        seed = args.seed
    torch.manual_seed(seed)
    print("Using seed:", seed)

    i = 0  # Iteration counter
    j = 0  # Zoom video frame counter
    p = 1  # Phrase counter
    smoother = 0  # Smoother counter
    this_video_frame = 0  # for video styling

    # Messing with learning rate / optimisers
    # variable_lr = args.step_size
    # optimiser_list = [['Adam',0.075],['AdamW',0.125],['Adagrad',0.2],['Adamax',0.125],['DiffGrad',0.075],['RAdam',0.125],['RMSprop',0.02]]

    # Do it
    try:
        with tqdm() as pbar:
            while True:
                # Change generated image
                if args.make_zoom_video:
                    if i % args.zoom_frequency == 0:
                        out = synth(z, gumbel)

                        # Save image
                        img = np.array(out.mul(255).clamp(0, 255)[0].cpu().detach().numpy().astype(np.uint8))[:, :, :]
                        img = np.transpose(img, (1, 2, 0))
                        imageio.imwrite("./steps/" + str(j) + ".png", np.array(img))

                        # Time to start zooming?
                        if args.zoom_start <= i:
                            # Convert z back into a Pil image
                            # pil_image = TF.to_pil_image(out[0].cpu())

                            # Convert NP to Pil image
                            pil_image = Image.fromarray(np.array(img).astype("uint8"), "RGB")

                            # Zoom
                            if args.zoom_scale != 1:
                                pil_image_zoom = zoom_at(pil_image, sideX / 2, sideY / 2, args.zoom_scale)
                            else:
                                pil_image_zoom = pil_image

                            # Shift - https://pillow.readthedocs.io/en/latest/reference/ImageChops.html
                            if args.zoom_shift_x or args.zoom_shift_y:
                                # This one wraps the image
                                pil_image_zoom = ImageChops.offset(pil_image_zoom, args.zoom_shift_x, args.zoom_shift_y)

                            # Convert image back to a tensor again
                            pil_tensor = TF.to_tensor(pil_image_zoom)

                            # Re-encode
                            z, *_ = model.encode(pil_tensor.to(device).unsqueeze(0) * 2 - 1)
                            z_orig = z.clone()
                            z.requires_grad_(True)

                            # Re-create optimiser
                            opt = get_opt(args.optimiser, args.step_size)

                        # Next
                        j += 1

                # Change text prompt
                if args.prompt_frequency > 0:
                    if i % args.prompt_frequency == 0 and i > 0:
                        # In case there aren't enough phrases, just loop
                        if p >= len(all_phrases):
                            p = 0

                        pMs = []
                        args.prompts = all_phrases[p]

                        # Show user we're changing prompt
                        print(args.prompts)

                        for prompt in args.prompts:
                            txt, weight, stop = split_prompt(prompt)
                            embed = perceptor.encode_text(clip.tokenize(txt).to(device)).float()
                            pMs.append(Prompt(embed, weight, stop).to(device))

                        """
                        # Smooth test
                        smoother = args.zoom_frequency * 15 # smoothing over x frames
                        variable_lr = args.step_size * 0.25
                        opt = get_opt(args.optimiser, variable_lr)
                        """

                        p += 1

                """
                if smoother > 0:
                    if smoother == 1:
                        opt = get_opt(args.optimiser, args.step_size)
                    smoother -= 1
                """

                """
                # Messing with learning rate / optimisers
                if i % 225 == 0 and i > 0:
                    variable_optimiser_item = random.choice(optimiser_list)
                    variable_optimiser = variable_optimiser_item[0]
                    variable_lr = variable_optimiser_item[1]
                    
                    opt = get_opt(variable_optimiser, variable_lr)
                    print("New opt: %s, lr= %f" %(variable_optimiser,variable_lr)) 
                """

                # Training time
                train(i, pMs, opt, perceptor, z, z_orig, z_min, z_max, gumbel, model, make_cutouts, args)
                
                # Ready to stop yet?
                if i == args.max_iterations:
                    if not args.video_style_dir:
                        # we're done
                        break
                    else:
                        if this_video_frame == (num_video_frames - 1):
                            # we're done
                            make_styled_video = True
                            break
                        else:
                            # Next video frame
                            this_video_frame += 1

                            # Reset the iteration count
                            i = -1
                            pbar.reset()

                            # Load the next frame, reset a few options - same filename, different directory
                            args.init_image = video_frame_list[this_video_frame]
                            print("Next frame: ", args.init_image)

                            if args.seed is None:
                                seed = torch.seed()
                            else:
                                seed = args.seed
                            torch.manual_seed(seed)
                            print("Seed: ", seed)

                            filename = os.path.basename(args.init_image)
                            args.output = os.path.join(cwd, "steps", filename)

                            # Load and resize image
                            img = Image.open(args.init_image)
                            pil_image = img.convert("RGB")
                            pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
                            pil_tensor = TF.to_tensor(pil_image)

                            # Re-encode
                            z, *_ = model.encode(pil_tensor.to(device).unsqueeze(0) * 2 - 1)
                            z_orig = z.clone()
                            z.requires_grad_(True)

                            # Re-create optimiser
                            opt = get_opt(args.optimiser, args.step_size)

                i += 1
                pbar.update()
    except KeyboardInterrupt:
        pass

    # All done :)

    # Video generation
    if args.make_video or args.make_zoom_video:
        init_frame = 1  # Initial video frame
        if args.make_zoom_video:
            last_frame = j
        else:
            last_frame = i  # This will raise an error if that number of frames does not exist.

        length = args.video_length  # Desired time of the video in seconds

        min_fps = 10
        max_fps = 60

        total_frames = last_frame - init_frame

        frames = []
        tqdm.write("Generating video...")
        for i in range(init_frame, last_frame):
            temp = Image.open("./steps/" + str(i) + ".png")
            keep = temp.copy()
            frames.append(keep)
            temp.close()

        if args.output_video_fps > 9:
            # Hardware encoding and video frame interpolation
            print("Creating interpolated frames...")
            ffmpeg_filter = f"minterpolate='mi_mode=mci:me=hexbs:me_mode=bidir:mc_mode=aobmc:vsbmc=1:mb_size=8:search_param=32:fps={args.output_video_fps}'"
            output_file = re.compile("\.png$").sub(".mp4", args.output)
            try:
                p = Popen(
                    [
                        "ffmpeg",
                        "-y",
                        "-f",
                        "image2pipe",
                        "-vcodec",
                        "png",
                        "-r",
                        str(args.input_video_fps),
                        "-i",
                        "-",
                        "-b:v",
                        "10M",
                        "-vcodec",
                        "h264_nvenc",
                        "-pix_fmt",
                        "yuv420p",
                        "-strict",
                        "-2",
                        "-filter:v",
                        f"{ffmpeg_filter}",
                        "-metadata",
                        f"comment={args.prompts}",
                        output_file,
                    ],
                    stdin=PIPE,
                )
            except FileNotFoundError:
                print("ffmpeg command failed - check your installation")
            for im in tqdm(frames):
                im.save(p.stdin, "PNG")
            p.stdin.close()
            p.wait()
        else:
            # CPU
            fps = np.clip(total_frames / length, min_fps, max_fps)
            output_file = re.compile("\.png$").sub(".mp4", args.output)
            try:
                p = Popen(
                    [
                        "ffmpeg",
                        "-y",
                        "-f",
                        "image2pipe",
                        "-vcodec",
                        "png",
                        "-r",
                        str(fps),
                        "-i",
                        "-",
                        "-vcodec",
                        "libx264",
                        "-r",
                        str(fps),
                        "-pix_fmt",
                        "yuv420p",
                        "-crf",
                        "17",
                        "-preset",
                        "veryslow",
                        "-metadata",
                        f"comment={args.prompts}",
                        output_file,
                    ],
                    stdin=PIPE,
                )
            except FileNotFoundError:
                print("ffmpeg command failed - check your installation")
            for im in tqdm(frames):
                im.save(p.stdin, "PNG")
            p.stdin.close()
            p.wait()
