import os
from pathlib import Path

import hydra
import safetensors
import torch
import gradio as gr
import numpy as np
from colorama import Fore
from jaxtyping import install_import_hook
from omegaconf import DictConfig, OmegaConf
from dacite import Config, from_dict
from typing import Literal
from einops import repeat
from jaxtyping import Float
from torch import Tensor
from PIL import Image
from videoscene.pipeline import VideoScenePipeline

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from noposplat.src.global_cfg import set_cfg
    from noposplat.src.loss import get_losses
    from noposplat.src.misc.LocalLogger import LocalLogger
    from noposplat.src.model.decoder import get_decoder
    from noposplat.src.model.encoder import get_encoder
    from noposplat.src.model.model_wrapper import ModelWrapper
    from noposplat.src.config import DecoderCfg, EncoderCfg
    from noposplat.src.dataset.types import BatchedExample, BatchedViews
    from noposplat.src.model.ply_export import export_ply


def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"

def load_image_as_numpy(path):
    return np.array(Image.open(path))


class APP:
    images: np.ndarray
    near: float = 0.1
    far: float = 100.0

    def __init__(self, model, video_model, prompt_embeds, output_dir, resolution=256):
        self.model = model
        self.video_model = video_model
        self.prompt_embeds = prompt_embeds
        self.output_dir = output_dir
        if type(resolution) == int:
            self.resolution = (resolution, resolution)
        else:
            self.resolution = resolution

    def launch(self):
        _DESCRIPTION = '''
<div style="display: flex; justify-content: center; align-items: center;">
    <div style="width: 100%; text-align: center; font-size: 30px;">
        <strong>VideoScene: Distilling Video Diffusion Model to Generate 3D Scenes in One Step</strong>
    </div>
</div> 
<p></p>

<div align="center">
    <a style="display:inline-block" href="https://arxiv.org/abs/2403.20309"><img src="https://img.shields.io/badge/ArXiv-2403.20309-b31b1b?logo=arxiv" alt='arxiv'></a>
    <a style="display:inline-block" href="https://hanyang-21.github.io/VideoScene/"><img src='https://img.shields.io/badge/Project-Website-green.svg'></a>&nbsp;
</div>
<p></p>

* Official demo of: [VideoScene: Distilling Video Diffusion Model to Generate 3D Scenes in One Step](https://hanyang-21.github.io/VideoScene/).
* Sparse-view examples for direct viewing: you can simply click the examples (in the bottom of the page), to quickly view the results on representative data.
'''

        with gr.Blocks() as app:
            description = gr.Markdown(_DESCRIPTION)
            with gr.Row():
                with gr.Column():
                    images = gr.Gallery(label="Input", type="numpy", columns=2)
                    apply = gr.Button(value="RUN")
                    example_names = ["room", "diningroom", "livingroom"]
                    
                    example_pairs = [[f"assets/{name}_1.png", f"assets/{name}_2.png"] for name in example_names]
                    gr.Examples(
                        examples=example_pairs,
                        inputs=[gr.Image(type="filepath", visible=False), gr.Image(type="filepath", visible=False)],
                        fn=lambda img1, img2: self.load_images_as_numpy(img1, img2),
                        outputs=images,
                        cache_examples=False,
                        label="Examples",
                        run_on_click=True,
                    )
                with gr.Column():
                    video = gr.Video(label="Video Output")
            
        
            images.upload(self.on_upload, images)
            apply.click(self.on_apply, [], [video])

        app.launch()

    def on_upload(self, inps: np.ndarray):
        images = []
        for (image, path) in inps:
            h, w = image.shape[:2]

            if image.shape[2] == 4: # Remove alpha channel if present.
                image = image[:, :, :3]

            min_dim = min(h, w)
            start_x = (w - min_dim) // 2
            start_y = (h - min_dim) // 2
            
            cropped_image = image[start_y:start_y + min_dim, start_x:start_x + min_dim]
            images.append(cropped_image)

        self.images = np.stack(images)

    def on_apply(self):
        assert self.images is not None, "Please upload two images first."
        assert len(self.images) == 2, "Please upload two images."
        images = torch.from_numpy(self.images).permute(0, 3, 1, 2).float().cuda() / 255.0
        images = torch.nn.functional.interpolate(images, size=self.resolution, mode='bilinear', align_corners=False)
        images = images.unsqueeze(0)

        cx, cy = 0.5, 0.5
        fx = fy = 0.8

        intrinsics = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).float().reshape(1, 1, 3, 3).repeat(1, 2, 1, 1).cuda()

        scale = 1
        (near,)=self.get_bound("near", images.shape[1]) / scale,
        (far,)=self.get_bound("far", images.shape[1]) / scale,
        batch = BatchedExample(
            context=BatchedViews(
                extrinsics=torch.randn(images.shape[0], images.shape[1], 4, 4, dtype=torch.float32),
                intrinsics=intrinsics,
                image=images,
                near=near.unsqueeze(0).float().cuda(),
                far=far.unsqueeze(0).float().cuda(),
            ),
            target=BatchedViews(
                extrinsics=None,
                intrinsics=None,
                image=None,
                near=None,
                far=None,
            )
        )
        
        with torch.no_grad() and torch.autocast(device_type="cuda"):
            images = self.model.predict(batch)
            
            torch.cuda.empty_cache()
            image_pair = torch.stack([images[0], images[-1]]).permute(0, 2, 3, 1)
            image_pair = image_pair.cpu().numpy()
            video_result_path = self.video_model.video_gen(f'{self.output_dir}/video/output_tmp.mp4', image_pair, self.prompt_embeds, noise_timestep=460)

            os.system(f"rm {self.output_dir}/video/output_tmp.mp4")
        return video_result_path

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    def load_images_as_numpy(self, img1_path, img2_path):
        inps = [load_image_as_numpy(img1_path), load_image_as_numpy(img2_path)]
        images = []
        for image in inps:
            h, w = image.shape[:2]

            if image.shape[2] == 4: # Remove alpha channel if present.
                image = image[:, :, :3]

            min_dim = min(h, w)
            start_x = (w - min_dim) // 2
            start_y = (h - min_dim) // 2
            
            cropped_image = image[start_y:start_y + min_dim, start_x:start_x + min_dim]
            images.append(cropped_image)

        self.images = np.stack(images)
        return inps


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="app",
)
def run(cfg_dict: DictConfig):
    set_cfg(cfg_dict)
    
    device = 'cuda'

    # Set up the output directory.
    if cfg_dict.output_dir is None:
        output_dir = Path(
            hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
        )
    else:  # for resuming
        output_dir = Path(cfg_dict.output_dir)
        os.makedirs(output_dir, exist_ok=True)
    print(cyan(f"Saving outputs to {output_dir}."))

    # Prepare the checkpoint for loading.
    recon_ckpt: str = cfg_dict.checkpointing.load
    pretrain_ckpt: str = cfg_dict.test.pretrain
    video_ckpt: str = cfg_dict.test.video

    torch.manual_seed(cfg_dict.seed)

    encoder_cfg: EncoderCfg = from_dict(data_class=EncoderCfg, data=OmegaConf.to_container(cfg_dict.model.encoder))

    encoder, encoder_visualizer = get_encoder(encoder_cfg)

    decoder_cfg: DecoderCfg = from_dict(data_class=DecoderCfg, data=OmegaConf.to_container(cfg_dict.model.decoder))

    decoder = get_decoder(decoder_cfg).to(device)

    model_wrapper = ModelWrapper(
        None,
        None,
        None,
        encoder,
        encoder_visualizer,
        decoder,
        None,
        None,
        None,
        output_dir
    ).to(device)

    # Reconstructing scene for 3D-aware prior
    state_dict_3d = torch.load(recon_ckpt, map_location="cpu")['state_dict']
    model_wrapper.load_state_dict(state_dict_3d)
    model_wrapper.to(device)
    
    # Generation video
    pipe = VideoScenePipeline.from_pretrained(
        pretrain_ckpt,
        torch_dtype=torch.bfloat16
    ).to('cuda')

    state_dict = safetensors.torch.load_file(video_ckpt)
    pipe.transformer.load_state_dict(state_dict, strict=False)
    
    prompt_embeds = torch.load("checkpoints/prompt_embeds.pt", map_location='cpu')

    # pipe.enable_sequential_cpu_offload()
    # pipe.vae.enable_tiling()
    # pipe.vae.enable_slicing()

    app = APP(model_wrapper, pipe, prompt_embeds, output_dir=output_dir, resolution=cfg_dict.resolution)

    app.launch()


if __name__ == "__main__":
    run()
