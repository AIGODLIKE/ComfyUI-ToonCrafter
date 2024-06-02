import os
import sys
import torch
import time
import logging as logger

from pathlib import Path
from contextlib import contextmanager, ExitStack
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download
from einops import repeat, rearrange
from torchvision import transforms
from pytorch_lightning import seed_everything
from platform import system

if system() == "Darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["XH_HOME"] = "~/.cache/huggingface"
os.environ["XFORMERS_FORCE_DISABLE_TRITON"] = "1"
# os.environ["no_proxy"] = "localhost, 127.0.0.1, ::1"

ROOT = Path(__file__).parent.joinpath("ToonCrafter")
sys.path.append(Path(__file__).parent.as_posix())
from ToonCrafter.utils.utils import instantiate_from_config
from ToonCrafter.scripts.evaluation.funcs import load_model_checkpoint, batch_ddim_sampling


class ToonCrafterNode:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "image2": ("IMAGE", ),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                # "clip": ("CLIP", ),
                "seed": ("INT", {"default": 123, "min": 0, "max": 0xffffffffffffffff}),
                "eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 15.0, "step": 0.1}),
                "cfg_scale": ("FLOAT", {"default": 7.5, "min": 1.0, "max": 15.0, "step": 0.5}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 60, "step": 1}),
                "frame_count": ("INT", {"default": 10, "min": 5, "max": 30, "step": 1}),
                "fps": ("INT", {"default": 8, "min": 1, "max": 60, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "get_image"

    OUTPUT_NODE = True

    CATEGORY = "ToonCrafter"

    def __init__(self, result_dir=ROOT.joinpath("tmp/"), gpu_num=1, resolution='320_512') -> None:
        h, w = resolution.split('_')
        self.resolution = int(h), int(w)
        self.download_model()

        self.result_dir = result_dir
        Path(self.result_dir).mkdir(parents=True, exist_ok=True)
        ckpt_path = ROOT.joinpath(f'checkpoints/tooncrafter_{w}_interp_v1', 'model.ckpt')
        config_file = ROOT.joinpath(f'configs/inference_{w}_v1.0.yaml')
        config = OmegaConf.load(config_file.as_posix())
        model_config = config.pop("model", OmegaConf.create())
        model_config['params']['unet_config']['params']['use_checkpoint'] = False
        model_list = []
        for gpu_id in range(gpu_num):
            model = instantiate_from_config(model_config)
            # model = model.cuda(gpu_id)
            logger.info(ckpt_path)
            assert ckpt_path.exists(), "Error: checkpoint Not Found!"
            model = load_model_checkpoint(model, ckpt_path.as_posix())
            model.eval()
            model_list.append(model)
        self.model_list = model_list
        self.save_fps = 8
        self.is_cuda = torch.cuda.is_available()
        self.is_mps = torch.backends.mps.is_available()

    @contextmanager
    def optional_autocast(device):
        try:
            with torch.autocast(device.type):
                yield
        except Exception as e:
            print(f"Autocast is not supported: {e}")
            yield

    def get_image(self, image: torch.Tensor, prompt, steps=50, cfg_scale=7.5, eta=1.0, frame_count=3, fps=8, seed=123, image2: torch.Tensor = None):
        self.save_fps = fps
        seed = seed % 4294967295
        seed_everything(seed)
        transform = transforms.Compose([
            transforms.Resize(min(self.resolution)),
            transforms.CenterCrop(self.resolution),
        ])
        if self.is_cuda:
            torch.cuda.empty_cache()
        elif self.is_mps:
            torch.mps.empty_cache()
        print('start:', prompt, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        start = time.time()
        gpu_id = 0
        if steps > 60:
            steps = 60
        model: torch.nn.Module = self.model_list[gpu_id]
        if self.is_cuda:
            model = model.cuda()
        elif self.is_mps:
            model = model.to('mps')
        batch_size = 1
        channels = model.model.diffusion_model.out_channels
        frames = model.temporal_length
        h, w = self.resolution[0] // 8, self.resolution[1] // 8
        noise_shape = [batch_size, channels, frames, h, w]

        # text cond
        with ExitStack() as stack:
            stack.enter_context(torch.no_grad())
            if self.is_cuda:
                stack.enter_context(torch.cuda.amp.autocast())
            # stack.enter_context(self.optional_autocast(device=model.device))
            text_emb = model.get_learned_conditioning([prompt])

            # img cond
            img_tensor = image[0].permute(2, 0, 1).float().to(model.device)
            img_tensor = (img_tensor - 0.5) * 2

            image_tensor_resized = transform(img_tensor)  # 3,h,w
            videos = image_tensor_resized.unsqueeze(0).unsqueeze(2)  # bc1hw

            # z = get_latent_z(model, videos) #bc,1,hw
            videos = repeat(videos, 'b c t h w -> b c (repeat t) h w', repeat=frames // 2)
            img_tensor2 = image2[0].permute(2, 0, 1).float().to(model.device)
            img_tensor2 = (img_tensor2 - 0.5) * 2
            image_tensor_resized2 = transform(img_tensor2)  # 3,h,w
            videos2 = image_tensor_resized2.unsqueeze(0).unsqueeze(2)  # bchw
            videos2 = repeat(videos2, 'b c t h w -> b c (repeat t) h w', repeat=frames // 2)

            videos = torch.cat([videos, videos2], dim=2)
            z, hs = self.get_latent_z_with_hidden_states(model, videos)

            img_tensor_repeat = torch.zeros_like(z)

            img_tensor_repeat[:, :, :1, :, :] = z[:, :, :1, :, :]
            img_tensor_repeat[:, :, -1:, :, :] = z[:, :, -1:, :, :]

            cond_images = model.embedder(img_tensor.unsqueeze(0))  # blc
            img_emb = model.image_proj_model(cond_images)

            imtext_cond = torch.cat([text_emb, img_emb], dim=1)

            fs = torch.tensor([frame_count], dtype=torch.long, device=model.device)
            cond = {"c_crossattn": [imtext_cond], "fs": fs, "c_concat": [img_tensor_repeat]}

            # inference
            batch_samples = batch_ddim_sampling(model, cond, noise_shape, n_samples=1, ddim_steps=steps, ddim_eta=eta, cfg_scale=cfg_scale, hs=hs)

            # remove the last frame
            if image2 is None:
                batch_samples = batch_samples[:, :, :, :-1, ...]
            # b,samples,c,t,h,w
            prompt_str = prompt.replace("/", "_slash_") if "/" in prompt else prompt
            prompt_str = prompt_str.replace(" ", "_") if " " in prompt else prompt_str
            prompt_str = prompt_str[:40]
            if len(prompt_str) == 0:
                prompt_str = 'empty_prompt'

        # save_videos(batch_samples, self.result_dir, filenames=[prompt_str], fps=self.save_fps)
        print(f"Saved in {prompt_str}. Time used: {(time.time() - start):.2f} seconds")
        try:
            # frame_count, width, height, channel
            batch_samples = batch_samples[0][0].permute(1, 2, 3, 0)
        except Exception as e:
            sys.stderr.write(e)
            return (None, )
        batch_samples = (batch_samples + 1.0) * 0.5
        model = model.cpu()
        return (batch_samples, )

    def download_model(self):
        REPO_ID = 'Doubiiu/ToonCrafter'
        filename_list = ['model.ckpt']
        model_dir = ROOT.joinpath('checkpoints/tooncrafter_' + str(self.resolution[1]) + '_interp_v1/')
        model_dir.mkdir(parents=True, exist_ok=True)
        for filename in filename_list:
            local_file = model_dir.joinpath(filename)
            if not local_file.exists():
                hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir=model_dir.as_posix(), local_dir_use_symlinks=False)

    def get_latent_z_with_hidden_states(self, model, videos):
        b, c, t, h, w = videos.shape
        x = rearrange(videos, 'b c t h w -> (b t) c h w')
        encoder_posterior, hidden_states = model.first_stage_model.encode(x, return_hidden_states=True)

        hidden_states_first_last = []
        # use only the first and last hidden states
        for hid in hidden_states:
            hid = rearrange(hid, '(b t) c h w -> b c t h w', t=t)
            hid_new = torch.cat([hid[:, :, 0:1], hid[:, :, -1:]], dim=2)
            hidden_states_first_last.append(hid_new)

        z = model.get_first_stage_encoding(encoder_posterior).detach()
        z = rearrange(z, '(b t) c h w -> b c t h w', b=b, t=t)
        return z, hidden_states_first_last


NODE_CLASS_MAPPINGS = {
    "ToonCrafterNode": ToonCrafterNode,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "ToonCrafterNode": "ToonCrafter",
}

WEB_DIRECTORY = "./"
