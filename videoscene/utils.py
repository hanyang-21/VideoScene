import os
from typing import Optional, Tuple
import torch 
from torchvision.transforms import InterpolationMode 
import numpy as np
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid
import torchvision.transforms as TT  
from torchvision import transforms
from torchvision.transforms.functional import center_crop, resize 
import decord
# import rp
# rp.git_import('CommonSource') #If missing, installs code from https://github.com/RyannDaGreat/CommonSource
# import rp.git.CommonSource.noise_warp as nw
# import einops

# def get_warp_noise(video_path, device, dtype):
#     FRAME = 2**-1 #We immediately resize the input frames by this factor, before calculating optical flow
#                   #The flow is calulated at (input size) × FRAME resolution.
#                   #Higher FLOW values result in slower optical flow calculation and higher intermediate noise resolution
#                   #Larger is not always better - watch the preview in Jupyter to see if it looks good!

#     FLOW = 2**3   #Then, we use bilinear interpolation to upscale the flow by this factor
#                     #We warp the noise at (input size) × FRAME × FLOW resolution
#                     #The noise is then downsampled back to (input size)
#                     #Higher FLOW values result in more temporally consistent noise warping at the cost of higher VRAM usage and slower inference time
#     LATENT = 8    #We further downsample the outputs by this amount - because 8 pixels wide corresponds to one latent wide in Stable Diffusion
#                     #The final output size is (input size) ÷ LATENT regardless of FRAME and FLOW

#     #LATENT = 1    #Uncomment this line for a prettier visualization! But for latent diffusion models, use LATENT=8

#     output_folder = "NoiseWarpOutputFolder"
#     video_path

#     if isinstance(video_path,str):
#         video=rp.load_video(video_path)

#     #Preprocess the video
#     video=rp.resize_list(video,length=49) #Stretch or squash video to 49 frames (CogVideoX's length)
#     video=rp.resize_images_to_hold(video,height=480,width=720)
#     video=rp.crop_images(video,height=480,width=720,origin='center') #Make the resolution 480x720 (CogVideoX's resolution)
#     video=rp.as_numpy_array(video)


#     #See this function's docstring for more information!
#     output = nw.get_noise_from_video(
#         video,
#         remove_background=False, #Set this to True to matte the foreground - and force the background to have no flow
#         visualize=True,          #Generates nice visualization videos and previews in Jupyter notebook
#         save_files=False,         #Set this to False if you just want the noises without saving to a numpy file
        
#         noise_channels=16,
#         output_folder=output_folder,
#         resize_frames=FRAME,
#         resize_flow=FLOW,
#         downscale_factor=round(FRAME * FLOW) * LATENT,
#     )
#     noise = torch.tensor(output['numpy_noises'])
#     noise = einops.rearrange(noise, 'F H W C -> F C H W')

#     def get_downtemp_noise(noise, noise_downtemp_interp):
#         assert noise_downtemp_interp in {'nearest', 'blend', 'blend_norm', 'randn'}, noise_downtemp_interp
#         if   noise_downtemp_interp == 'nearest'    : return                  rp.resize_list(noise, 13)
#         # elif noise_downtemp_interp == 'blend'      : return                   downsamp_mean(noise, 13)
#         # elif noise_downtemp_interp == 'blend_norm' : return normalized_noises(downsamp_mean(noise, 13))
#         elif noise_downtemp_interp == 'randn'      : return torch.randn_like(rp.resize_list(noise, 13)) #Basically no warped noise, just r
#         else: assert False, 'impossible'

#     downtemp_noise = get_downtemp_noise(
#         noise,
#         noise_downtemp_interp='nearest',
#     )
#     downtemp_noise = downtemp_noise[None]
#     downtemp_noise = nw.mix_new_noise(downtemp_noise, .5)
#     downtemp_noise = downtemp_noise.to(
#         device, dtype=dtype
#     )
#     return downtemp_noise # [1, F, C, H, W]

def prepare_rotary_positional_embeddings(
    height: int,
    width: int,
    num_frames: int,
    vae_scale_factor_spatial: int = 8,
    patch_size: int = 2,
    attention_head_dim: int = 64,
    device: Optional[torch.device] = None,
    base_height: int = 480,
    base_width: int = 720,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grid_height = height // (vae_scale_factor_spatial * patch_size)
    grid_width = width // (vae_scale_factor_spatial * patch_size)
    base_size_width = base_width // (vae_scale_factor_spatial * patch_size)
    base_size_height = base_height // (vae_scale_factor_spatial * patch_size)

    grid_crops_coords = get_resize_crop_region_for_grid((grid_height, grid_width), base_size_width, base_size_height)
    freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
        embed_dim=attention_head_dim,
        crops_coords=grid_crops_coords,
        grid_size=(grid_height, grid_width),
        temporal_size=num_frames,
    )

    freqs_cos = freqs_cos.to(device=device)
    freqs_sin = freqs_sin.to(device=device)
    return freqs_cos, freqs_sin

def compute_prompt_embeds(
    text_encoder,
    text_input_ids,
    device=None,
    dtype=None,
    num_videos_per_prompt=1,
):  
    batch_size = text_input_ids.size(0)
    print(text_input_ids)
    with torch.no_grad():
        prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    print(prompt_embeds)
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)
    return prompt_embeds

def _resize_for_rectangle_crop(arr):
    image_size = 480, 720
    reshape_mode = 'center'
    if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
        arr = resize(
            arr,
            size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
            interpolation=InterpolationMode.BICUBIC,
        )
    else:
        arr = resize(
            arr,
            size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
            interpolation=InterpolationMode.BICUBIC,
        )

    h, w = arr.shape[2], arr.shape[3]
    arr = arr.squeeze(0)

    delta_h = h - image_size[0]
    delta_w = w - image_size[1]

    if reshape_mode == "random" or reshape_mode == "none":
        top = np.random.randint(0, delta_h + 1)
        left = np.random.randint(0, delta_w + 1)
    elif reshape_mode == "center":
        top, left = delta_h // 2, delta_w // 2
    else:
        raise NotImplementedError
    arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
    return arr
    
def process_video(video_path):
    while True: 
        video_reader = decord.VideoReader(video_path, width=720, height=480)
        video_num_frames = len(video_reader) 
        # print(video_num_frames, video_reader.get_avg_fps()) 
        if 2 * 49 > video_num_frames: 
            stripe = 1
        else:
            stripe = 2

        random_range = video_num_frames - stripe * 49 - 1
        random_range = max(1, random_range)
        start_frame = 0 # random.randint(1, random_range) if random_range > 0 else 1
        
        indices = list(range(start_frame, start_frame + stripe * 49, stripe)) # (end_frame - start_frame) // 49))
        frames = video_reader.get_batch(indices).asnumpy()

        # Ensure that we don't go over the limit
        frames = torch.from_numpy(frames)[0:49]
        selected_num_frames = frames.shape[0]

        # Choose first (4k + 1) frames as this is how many is required by the VAE
        remainder = (3 + (selected_num_frames % 4)) % 4
        if remainder != 0:
            frames = frames[:-remainder]
        selected_num_frames = frames.shape[0]

        assert (selected_num_frames - 1) % 4 == 0 
        if selected_num_frames == 49: 
            break 
        else:
            continue 

    # Training transforms
    # frames = (frames - 127.5) / 127.5
    frames = frames.permute(0, 3, 1, 2).contiguous()  # [F, C, H, W]
    frames = _resize_for_rectangle_crop(frames) 
    video_transforms = transforms.Compose(
            [
                transforms.Lambda(lambda x: x),
                transforms.Lambda(lambda x: x / 255.0),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
    # print(frames[0])
    # frames = torch.stack([video_transforms(frame) for frame in frames], dim=0)
    frames = torch.stack([frame for frame in frames], dim=0)
    return frames.contiguous()
