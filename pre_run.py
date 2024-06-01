import sys
import argparse
import os
from pathlib import Path
sys.path.append(Path(__file__).parent.as_posix())
from scripts.evaluation.inference import seed_everything, run_inference, get_parser


def run():
    old_dir = os.getcwd()
    os.chdir(Path(__file__).parent.as_posix())
    parser = get_parser()
    ckpt = "checkpoints/tooncrafter_512_interp_v1/model.ckpt"
    config = "configs/inference_512_v1.0.yaml"

    prompt_dir = "prompts/512_interp/"
    res_dir = "results"

    FS = 10  # This model adopts FPS=5, range recommended: 5-30 (smaller value -> larger motion)

    seed = 123
    name = f"tooncrafter_512_interp_seed{seed}"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    namespace = argparse.Namespace(
        seed=123,
        ckpt_path=ckpt,
        config=config,
        savedir=f"{res_dir}/{name}",
        n_samples=1,
        bs=1,
        height=320,
        width=512,
        unconditional_guidance_scale=7.5,
        ddim_steps=50,
        ddim_eta=1.0,
        prompt_dir=prompt_dir,
        text_input=True,
        video_length=16,
        frame_stride=FS,
        timestep_spacing='uniform_trailing',
        guidance_rescale=0.7,
        perframe_ae=True,
        interp=True
    )
    args = parser.parse_args(args=[], namespace=namespace)
    seed_everything(args.seed)
    rank, gpu_num = 0, 1
    run_inference(args, gpu_num, rank)
    os.chdir(old_dir)


if __name__ == "__main__":
    run()
