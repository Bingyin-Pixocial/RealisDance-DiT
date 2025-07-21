import argparse
import glob
import numpy as np
import os
import torch

from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video
from PIL import Image
from src.pipelines.rd_dit_pipeline import RealisDanceDiTPipeline
from src.utils.dist_utils import hook_for_multi_gpu_inference, init_dist, is_main_process, set_seed
from transformers import CLIPVisionModel

import decord

decord.bridge.set_bridge("torch")


def is_image(path):
    valid_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
    return path.lower().endswith(valid_extensions)


def load_image(path, div=127.5, add=-1):
    # read image
    image = torch.from_numpy(np.array(Image.open(path).convert("RGB")))
    # [0, 255] -> [-1, 1]
    image = image / div + add
    # H W C -> B C F H W
    image = image.permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
    return image


def load_video(
    path,
    fps=16,
    start_index=0,
    num_frames=81,
    div=127.5,
    add=-1,
):
    video_reader = decord.VideoReader(path)
    video_length = len(video_reader)
    ori_fps = video_reader.get_avg_fps()
    print(f"ori_fps: {ori_fps}")
    normed_video_length = max(round(video_length / ori_fps * fps), num_frames)
    batch_index_all = np.linspace(0, video_length - 1, normed_video_length).round().astype(int).tolist()
    batch_index = batch_index_all[start_index:start_index + num_frames]
    video = video_reader.get_batch(batch_index).permute(3, 0, 1, 2).unsqueeze(0).contiguous()
    del video_reader
    video = video / div + add
    return video  # 1 C T H W


def main():
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', type=str, default=None, help='path to reference image.')
    parser.add_argument('--smpl', type=str, default=None, help='Path to smpl video.')
    parser.add_argument('--hamer', type=str, default=None, help='Path to hamer video.')
    parser.add_argument('--prompt', type=str, default=None, help='Prompt for video.')
    parser.add_argument('--root', type=str, default=None, help='Root path for batch inference.')
    parser.add_argument('--save-dir', type=str, default="./output", help='Path to output folder.')
    parser.add_argument('--ckpt', type=str, default="./pretrained_models", help='Path to checkpoint folder.')
    parser.add_argument('--max-res', type=int, default=768 * 768, help='Resolution of the generated video.')
    parser.add_argument('--num-frames', type=int, default=81, help='Number of the generated video frames.')
    parser.add_argument('--fps', type=int, default=16, help='FPS of the generated video.')
    parser.add_argument('--seed', type=int, default=1024, help='The generation seed.')
    parser.add_argument('--save-gpu-memory', action='store_true', help='Save GPU memory, but will be super slow.')
    parser.add_argument(
        '--multi-gpu', action='store_true', help='Enable FSDP and Sequential parallel for multi-GPU inference.',
    )
    parser.add_argument(
        '--enable-teacache', action='store_true',
        help='Enable teacache to accelerate inference. Note that enabling teacache may hurt generation quality.',
    )
    args = parser.parse_args()

    # assign args
    ref_path = args.ref
    smpl_path = args.smpl
    hamer_path = args.hamer
    prompt = args.prompt
    root = args.root
    save_dir = args.save_dir
    ckpt = args.ckpt
    max_res = args.max_res
    num_frames = args.num_frames
    seed = args.seed
    save_gpu_memory = args.save_gpu_memory
    multi_gpu = args.multi_gpu
    enable_teacache = args.enable_teacache
    fps = args.fps
    os.makedirs(save_dir, exist_ok=True)

    # check args
    if root is None and (ref_path is None or smpl_path is None or hamer_path is None):
        raise ValueError("`root` and `ref` / `smpl` / `hamer` cannot be None at the same time.")
    elif root is not None and (ref_path is not None or smpl_path is not None or hamer_path is not None):
        print("WARNING: Will not use `ref` / `smpl` / `hamer` when `root` is not None.")
    if save_gpu_memory and multi_gpu:
        raise ValueError("`--multi-gpu` and `--save-gpu-memory` cannot be set at the same time.")

    # init dist and set seed
    if multi_gpu:
        init_dist()
    set_seed(seed)

    # load model
    model_id = ckpt
    image_encoder = CLIPVisionModel.from_pretrained(
        model_id, subfolder="image_encoder", torch_dtype=torch.float32
    )
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = RealisDanceDiTPipeline.from_pretrained(
        model_id, vae=vae, image_encoder=image_encoder, torch_dtype=torch.bfloat16
    )
    if save_gpu_memory:
        print("WARNING: Enable sequential cpu offload which will be super slow.")
        pipe.enable_sequential_cpu_offload()
    elif multi_gpu:
        pipe = hook_for_multi_gpu_inference(pipe)
    else:
        pipe.enable_model_cpu_offload()

    # negative_prompt = "细节模糊不清，字幕，作品，画作，画面，静止，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，杂乱的背景，三条腿，背景人很多，倒着走"
    negative_prompt = ""

    # inference
    if root is not None:  # batch inference
        for ref_path in glob.glob(os.path.join(root, "ref", "*")):
            if not is_image(ref_path):
                continue

            # path process
            vid = os.path.splitext(os.path.basename(ref_path))[0]
            output_path = os.path.join(save_dir, f"{vid}.mp4")
            smpl_path = os.path.join(root, "smpl", f"{vid}.mp4")
            hamer_path = os.path.join(root, "hamer", f"{vid}.mp4")
            prompt_path = os.path.join(root, "prompt", f"{vid}.txt")

            # prompt process
            prompt = ""
            with open(prompt_path, 'r', encoding='utf-8') as file:
                for l in file.readlines():
                    prompt += l.strip()

            
            # prepare inputs, inference, and save
            ref_image = load_image(ref_path)
            smpl = load_video(smpl_path, fps=fps, num_frames=num_frames)
            hamer = load_video(hamer_path, fps=fps, num_frames=num_frames)
            output = pipe(
                image=ref_image,
                num_frames=num_frames,
                smpl=smpl,
                hamer=hamer,
                prompt=prompt,
                negative_prompt=negative_prompt,
                max_resolution=max_res,
                enable_teacache=enable_teacache,
            ).frames[0]
            if is_main_process():
                export_to_video(output, output_path, fps=fps)
    else:  # single sample inference
        # path process
        print("single sample inference")
        vid = os.path.splitext(os.path.basename(ref_path))[0]
        pose_id = os.path.splitext(os.path.basename(smpl_path))[0]
        output_path = os.path.join(save_dir, f"{vid}_{pose_id}.mp4")

        # prepare inputs, inference, and save
        ref_image = load_image(ref_path)
        print(f"ref_image shape: {ref_image.shape}")
        smpl = load_video(smpl_path, fps=fps, num_frames=num_frames)
        print(f"smpl shape: {smpl.shape}")
        hamer = load_video(hamer_path, fps=fps, num_frames=num_frames)
        print(f"hamer shape: {hamer.shape}")
        output = pipe(
            image=ref_image,
            num_frames=num_frames,
            smpl=smpl,
            hamer=hamer,
            prompt=prompt,
            negative_prompt=negative_prompt,
            max_resolution=max_res,
            enable_teacache=enable_teacache,
        ).frames[0]
        if is_main_process():
            export_to_video(output, output_path, fps=fps)


if __name__ == "__main__":
    main()
