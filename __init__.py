import os
import sys
import torch
import time
import logging as logger

from functools import cache
from pathlib import Path
from contextlib import contextmanager, ExitStack
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download
from einops import repeat, rearrange
from torchvision import transforms
from pytorch_lightning import seed_everything
from platform import system
from comfy import model_management as mm
from comfy.utils import ProgressBar

if system() == "Darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["HF_HOME"] = "~/.cache/huggingface"
os.environ["XFORMERS_FORCE_DISABLE_TRITON"] = "1"
USER_DEF_CLIP = Path(__file__).parent.joinpath("models/open_clip_pytorch_model.bin")
if USER_DEF_CLIP.exists():
    os.environ["USER_DEF_CLIP"] = USER_DEF_CLIP.as_posix()
# os.environ["no_proxy"] = "localhost, 127.0.0.1, ::1"
ROOT = Path(__file__).parent.joinpath("ToonCrafter")
sys.path.append(Path(__file__).parent.as_posix())
from ToonCrafter.utils.utils import instantiate_from_config
from ToonCrafter.scripts.evaluation.funcs import load_model_checkpoint, batch_ddim_sampling


@cache
def get_models(root: Path = ROOT.joinpath("checkpoints")):
    ckpts = []
    files = []
    for ext in ['ckpt', 'pt', 'bin', 'pth', 'safetensors', 'pkl']:
        files.extend(root.rglob(f"*.{ext}"))
    for file in files:
        ckpts.append(file.relative_to(root).as_posix())
    return sorted(ckpts)


class ToonCrafterNode:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "image2": ("IMAGE", ),
                "ckpt_name": (get_models(), ),
                "vram_opt_strategy": (["none", "low"], ),
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

    def init(self, ckpt_name="", result_dir=ROOT.joinpath("tmp/"), gpu_num=1, resolution='320_512') -> None:
        h, w = resolution.split('_')
        self.resolution = int(h), int(w)
        # self.download_model()

        self.result_dir = result_dir
        Path(self.result_dir).mkdir(parents=True, exist_ok=True)
        ckpt_path = ROOT.joinpath("checkpoints", ckpt_name)
        if not ckpt_path.exists():
            ckpt_path = ROOT.joinpath(f'checkpoints/tooncrafter_{w}_interp_v1', 'model.ckpt')
        if not ckpt_path.exists():
            raise Exception(f"ToonCrafterNode Error: {ckpt_path} Not Found!")
        config_file = ROOT.joinpath(f'configs/inference_{w}_v1.0.yaml')
        config = OmegaConf.load(config_file.as_posix())
        model_config = config.pop("model", OmegaConf.create())
        model_config['params']['unet_config']['params']['use_checkpoint'] = False
        model_list = []
        # mm.unload_all_models()
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
        self.is_cpu = torch.cpu.is_available()

    @contextmanager
    def optional_autocast(device):
        try:
            with torch.autocast(device.type):
                yield
        except Exception as e:
            print(f"Autocast is not supported: {e}")
            yield

    def get_image(self, image: torch.Tensor, ckpt_name, vram_opt_strategy, prompt, steps=50, cfg_scale=7.5, eta=1.0, frame_count=3, fps=8, seed=123, image2: torch.Tensor = None):
        os.environ["TOON_MEM_STRATEGY"] = vram_opt_strategy
        self.init(ckpt_name=ckpt_name)
        self.save_fps = fps
        seed = seed % 4294967295
        seed_everything(seed)
        transform = transforms.Compose([
            transforms.Resize(min(self.resolution)),
            transforms.CenterCrop(self.resolution),
        ])
        mm.soft_empty_cache()
        print('start:', prompt, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        start = time.time()
        gpu_id = 0
        if steps > 60:
            steps = 60
        model: torch.nn.Module = self.model_list[gpu_id]
        half = mm.should_use_bf16() or mm.should_use_fp16() or vram_opt_strategy == "low"
        if half:
            model = model.half()
            image = image.half()
            image2 = image2.half()
        if self.is_cuda:
            model = model.to('cuda')
        elif self.is_mps:
            model = model.to('mps')
        elif self.is_cpu:
            model = model.to('cpu')
        batch_size = 1
        channels = model.model.diffusion_model.out_channels
        frames = model.temporal_length
        h, w = self.resolution[0] // 8, self.resolution[1] // 8
        noise_shape = [batch_size, channels, frames, h, w]
        pbar = ProgressBar(steps)
        # text cond
        with ExitStack() as stack:
            stack.enter_context(torch.no_grad())
            if self.is_cuda:
                stack.enter_context(torch.cuda.amp.autocast())
            # stack.enter_context(self.optional_autocast(device=model.device))
            text_emb = model.get_learned_conditioning([prompt])
            model.cond_stage_model.to("cpu")
            # img cond
            img_tensor = image[0].permute(2, 0, 1).to(model.device)
            img_tensor = (img_tensor - 0.5) * 2

            image_tensor_resized = transform(img_tensor)  # 3,h,w
            videos = image_tensor_resized.unsqueeze(0).unsqueeze(2)  # bc1hw

            # z = get_latent_z(model, videos) #bc,1,hw
            videos = repeat(videos, 'b c t h w -> b c (repeat t) h w', repeat=frames // 2)
            img_tensor2 = image2[0].permute(2, 0, 1).to(model.device)
            img_tensor2 = (img_tensor2 - 0.5) * 2
            image_tensor_resized2 = transform(img_tensor2)  # 3,h,w
            videos2 = image_tensor_resized2.unsqueeze(0).unsqueeze(2)  # bchw
            videos2 = repeat(videos2, 'b c t h w -> b c (repeat t) h w', repeat=frames // 2)

            videos = torch.cat([videos, videos2], dim=2)
            # v10 = torch.mps.driver_allocated_memory() / 1024**3
            mm.soft_empty_cache()
            # v11 = torch.mps.driver_allocated_memory() / 1024**3
            z, hs = self.get_latent_z_with_hidden_states(model, videos)
            model.cond_stage_model.to(model.device)
            # v20 = torch.mps.driver_allocated_memory() / 1024**3
            mm.soft_empty_cache()
            # v21 = torch.mps.driver_allocated_memory() / 1024**3

            img_tensor_repeat = torch.zeros_like(z).to(dtype=model.dtype)

            img_tensor_repeat[:, :, :1, :, :] = z[:, :, :1, :, :]
            img_tensor_repeat[:, :, -1:, :, :] = z[:, :, -1:, :, :]

            cond_images = model.embedder(img_tensor.unsqueeze(0))  # blc
            img_emb = model.image_proj_model(cond_images)

            imtext_cond = torch.cat([text_emb, img_emb], dim=1)

            del cond_images, text_emb, img_emb, videos, videos2, image_tensor_resized2, img_tensor2, image_tensor_resized, image
            fs = torch.tensor([frame_count], dtype=torch.long, device=model.device)
            cond = {"c_crossattn": [imtext_cond], "fs": fs, "c_concat": [img_tensor_repeat]}

            def cb(step):
                print(f"step: {step}", end='\r')
                pbar.update_absolute(step + 1)

            mm.soft_empty_cache()
            # inference
            batch_samples = batch_ddim_sampling(model, cond, noise_shape, n_samples=1, ddim_steps=steps, ddim_eta=eta, cfg_scale=cfg_scale, hs=hs, callback=cb)

            # remove the last frame
            if image2 is None:
                batch_samples = batch_samples[:, :, :, :-1, ...]
            # b,samples,c,t,h,w
            prompt_str = prompt.replace("/", "_slash_") if "/" in prompt else prompt
            prompt_str = prompt_str.replace(" ", "_") if " " in prompt else prompt_str
            prompt_str = prompt_str[:40]
            if len(prompt_str) == 0:
                prompt_str = 'empty_prompt'

        # self.save_videos(batch_samples, self.result_dir, filenames=[prompt_str], fps=self.save_fps)
        print(f"Saved in {prompt_str}. Time used: {(time.time() - start):.2f} seconds")
        try:
            # frame_count, width, height, channel
            batch_samples = batch_samples[0][0].permute(1, 2, 3, 0)
            if half:
                batch_samples = batch_samples.to(dtype=torch.float32)
        except Exception as e:
            sys.stderr.write(f"{e}\n")
            return (None, )
        batch_samples = (batch_samples + 1.0) * 0.5
        mm.soft_empty_cache()
        model = model.cpu()
        return (batch_samples, )

    def save_videos(self, batch_tensors, savedir, filenames, fps=10):
        import torchvision
        # b,samples,c,t,h,w
        n_samples = batch_tensors.shape[1]
        for idx, vid_tensor in enumerate(batch_tensors):
            video = vid_tensor.detach().cpu()
            video = torch.clamp(video.float(), -1., 1.)
            video = video.permute(2, 0, 1, 3, 4)  # t,n,c,h,w
            frame_grids = [torchvision.utils.make_grid(framesheet, nrow=int(n_samples)) for framesheet in video]  # [3, 1*h, n*w]
            grid = torch.stack(frame_grids, dim=0)  # stack in temporal dim [t, 3, n*h, w]
            grid = (grid + 1.0) / 2.0
            grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)
            savepath = os.path.join(savedir, f"{filenames[idx]}.mp4")
            torchvision.io.write_video(savepath, grid, fps=fps, video_codec='h264', options={'crf': '10'})
            
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
