# Copyright 2025 The RealisDance-DiT Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import html
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import regex as re
import torch
import torch.nn.functional as F
from einops import rearrange
from transformers import AutoTokenizer, CLIPVisionModel, UMT5EncoderModel
from transformers.image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.loaders import WanLoraLoaderMixin
from diffusers.models import AutoencoderKLWan
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import is_ftfy_available, is_torch_xla_available, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor

from .rd_dit_pipeline import RealisDanceDiTPipeline, prompt_clean, retrieve_latents

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if is_ftfy_available():
    import ftfy

EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> import torch
        >>> from diffusers import AutoencoderKLWan
        >>> from transformers import CLIPVisionModel
        >>> from src.pipelines.rd_dit_pipeline_with_pose import RealisDanceDiTPipelineWithPose

        >>> model_id = "theFoxofSky/RealisDance-DiT"
        >>> image_encoder = CLIPVisionModel.from_pretrained(
        ...     model_id, subfolder="image_encoder", torch_dtype=torch.float32
        ... )
        >>> vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
        >>> pipe = RealisDanceDiTPipelineWithPose.from_pretrained(
        ...     model_id, vae=vae, image_encoder=image_encoder, torch_dtype=torch.bfloat16
        ... )
        >>> pipe.enable_model_cpu_offload()
        
        >>> # prepare ref_image, smpl, hamer, pose, all of them are in shape B C F H W, in range [-1, 1]
        
        >>> max_res = 768*768
        >>> output = pipe(
        ...     image=ref_image,
        ...     smpl=smpl,
        ...     hamer=hamer,
        ...     pose=pose,  # Optional: direct pose input
        ...     prompt=prompt,
        ...     max_resolution=max_res,
        ... ).frames[0]
        >>> export_to_video(output, "output.mp4", fps=16)
        ```
"""


class RealisDanceDiTPipelineWithPose(RealisDanceDiTPipeline):
    r"""
    Extended Pipeline for RealisDance-DiT that accepts pose information as a direct parameter.
    
    This pipeline extends the original RealisDanceDiTPipeline to support direct pose input
    while maintaining all the original functionality. When pose is provided, it will be used
    directly; otherwise, it falls back to the original behavior of combining smpl and hamer.

    Args:
        tokenizer ([`T5Tokenizer`]):
            Tokenizer from [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5Tokenizer),
            specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
        text_encoder ([`T5EncoderModel`]):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
        image_encoder ([`CLIPVisionModel`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPVisionModel), specifically
            the
            [clip-vit-huge-patch14](https://github.com/mlfoundations/open_clip/blob/main/docs/PRETRAINED.md#vit-h14-xlm-roberta-large)
            variant.
        transformer ([`RealisDanceDiT`]):
            Conditional Transformer to denoise the input latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKLWan`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
    """

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        image,
        smpl,
        hamer,
        pose=None,  # Added pose parameter
        height=None,
        width=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        # Call parent check_inputs first
        super().check_inputs(
            prompt,
            negative_prompt,
            image,
            smpl,
            hamer,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
        )
        
        # Additional check for pose if provided
        if pose is not None and not isinstance(pose, torch.Tensor):
            raise ValueError(f"`pose` has to be of type `torch.Tensor` but is {type(pose)}")

    def prepare_latents_with_pose(
        self,
        image: torch.Tensor,
        smpl: torch.Tensor,
        hamer: torch.Tensor,
        pose: Optional[torch.Tensor] = None,  # Added pose parameter
        batch_size: int = 1,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare latents with optional direct pose input.
        
        If pose is provided, it will be used directly; otherwise, it falls back to
        the original behavior of combining smpl and hamer.
        """
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial

        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        image = image.to(device=device, dtype=dtype)
        video_condition = torch.zeros(
            image.shape[0], image.shape[1], num_frames, height, width
        )
        video_condition = video_condition.to(device=device, dtype=dtype)
        smpl = smpl.to(device=device, dtype=dtype)
        hamer = hamer.to(device=device, dtype=dtype)

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )

        if isinstance(generator, list):
            def vae_encode_1(x):
                latent_x = [
                    retrieve_latents(self.vae.encode(x), sample_mode="argmax") for _ in generator
                ]
                return (torch.cat(latent_x) - latents_mean) * latents_std

            latent_condition = vae_encode_1(video_condition)
            latent_smpl = vae_encode_1(smpl)
            latent_hamer = vae_encode_1(hamer)
            latent_ref = vae_encode_1(image)
            if self.do_classifier_free_guidance:
                latent_null_ref = vae_encode_1(torch.zeros_like(image))
            else:
                latent_null_ref = None
        else:
            def vae_encode_2(x):
                latent_x = retrieve_latents(self.vae.encode(x), sample_mode="argmax")
                return (latent_x.repeat(batch_size, 1, 1, 1, 1) - latents_mean) * latents_std

            latent_condition = vae_encode_2(video_condition)
            latent_smpl = vae_encode_2(smpl)
            latent_hamer = vae_encode_2(hamer)
            latent_ref = vae_encode_2(image)
            if self.do_classifier_free_guidance:
                latent_null_ref = vae_encode_2(torch.zeros_like(image))
            else:
                latent_null_ref = None

        mask_lat_size = torch.zeros(batch_size, 4, num_latent_frames, latent_height, latent_width)
        mask_lat_size = mask_lat_size.to(latent_condition.device)
        latent_i2v_condition = torch.cat(
            [mask_lat_size, latent_condition], dim=1
        )

        # Handle pose processing
        if pose is not None:
            # Use provided pose directly
            pose = pose.to(device=device, dtype=dtype)
            if isinstance(generator, list):
                latent_external_pose = vae_encode_1(pose)
            else:
                latent_external_pose = vae_encode_2(pose)
            
            # Concatenate external pose with smpl and hamer for comprehensive pose information
            # This provides the most complete pose representation: external_pose + smpl + hamer
            latent_pose = torch.cat((latent_external_pose, latent_smpl, latent_hamer), dim=1)
            
            print(f"Constructed latent_pose with {latent_pose.shape[1]} channels: external_pose({latent_external_pose.shape[1]}) + smpl({latent_smpl.shape[1]}) + hamer({latent_hamer.shape[1]})")
        else:
            # Fall back to original behavior: combine smpl and hamer
            latent_pose = torch.cat((latent_smpl, latent_hamer), dim=1)
            print(f"Using original pose construction: smpl({latent_smpl.shape[1]}) + hamer({latent_hamer.shape[1]}) = {latent_pose.shape[1]} channels")

        return latents, latent_i2v_condition, latent_pose, latent_ref, latent_null_ref

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        image: torch.Tensor,
        smpl: torch.Tensor,
        hamer: torch.Tensor,
        pose: Optional[torch.Tensor] = None,  # Added pose parameter
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        max_resolution: int = 768 * 768,
        num_frames: int = 81,
        num_inference_steps: int = 40,
        guidance_scale: float = 2.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        enable_teacache: bool = False,
        teacache_thresh: float = 0.2,
        use_timestep_proj: bool = True,
    ):
        r"""
        The call function to the pipeline for generation with optional pose input.

        Args:
            image (`torch.Tensor`):
                The input image to condition the generation on. Must be a `torch.Tensor`
                with shape B C 1 H W and range [-1, 1].
            smpl (`torch.Tensor`):
                The input smpl video to condition the generation on. Must be a `torch.Tensor`
                with shape B C T H W and range [-1, 1].
            hamer (`torch.Tensor`):
                The input hamer video to condition the generation on. Must be a `torch.Tensor`
                with shape B C T H W and range [-1, 1].
            pose (`torch.Tensor`, *optional*):
                The input pose video to condition the generation on. Must be a `torch.Tensor`
                with shape B C T H W and range [-1, 1]. If not provided, pose will be
                automatically generated from smpl and hamer.
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            height (`int`, defaults to `480`):
                The height of the generated video.
            width (`int`, defaults to `832`):
                The width of the generated video.
            num_frames (`int`, defaults to `81`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, defaults to `5.0`):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`WanPipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, *optional*, defaults to `512`):
                The maximum sequence length of the prompt.
            enable_teacache (`bool`, *optional*, defaults to False):
                Whether to use teacache to accelerate inference. Note that enabling teacache will hurt generation
                quality.
            teacache_thresh (`float`, *optional*, defaults to 0.2):
                Threshold for teacache. Higher speedup will cause to worse quality.
            use_timestep_proj (`bool`, *optional*, defaults to True):
                Whether to use timestep_proj or temb.
        Examples:

        Returns:
            [`~WanPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`WanPipelineOutput`] is returned, otherwise a `tuple` is returned where
                the first element is a list with the generated images and the second element is a list of `bool`s
                indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            negative_prompt,
            image,
            smpl,
            hamer,
            pose,  # Pass pose to check_inputs
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
        )

        if height is None or width is None:
            smpl_height, smpl_width = smpl.shape[-2:]
            ratio = (max_resolution / (smpl_height * smpl_width)) ** 0.5
            scale_factor_h = self.vae_scale_factor_spatial * self.transformer.config.patch_size[1]
            height = round(smpl_height * ratio / scale_factor_h) * scale_factor_h
            scale_factor_w = self.vae_scale_factor_spatial * self.transformer.config.patch_size[2]
            width = round(smpl_width * ratio / scale_factor_w) * scale_factor_w

        if num_frames % self.vae_scale_factor_temporal != 1:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        # Encode image embedding
        transformer_dtype = self.transformer.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        image_embeds = self.encode_image(image, device)
        image_embeds = image_embeds.repeat(batch_size, 1, 1)
        image_embeds = image_embeds.to(transformer_dtype)
        if self.do_classifier_free_guidance:
            null_image_embeds = self.encode_image(torch.zeros_like(image), device)
            null_image_embeds = null_image_embeds.repeat(batch_size, 1, 1)
            null_image_embeds = null_image_embeds.to(transformer_dtype)
        else:
            null_image_embeds = None

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables with pose support
        num_channels_latents = self.vae.config.z_dim
        image = self.process_shape(image, height, width, resize_type="max_resolution").to(device, dtype=torch.float32)
        smpl = self.process_shape(smpl, height, width, resize_type="resize_crop").to(device, dtype=torch.float32)
        hamer = self.process_shape(hamer, height, width, resize_type="resize_crop").to(device, dtype=torch.float32)
        
        # Process pose if provided
        if pose is not None:
            pose = self.process_shape(pose, height, width, resize_type="resize_crop").to(device, dtype=torch.float32)
        
        latents, i2v_condition, pose_condition, ref_condition, null_ref_condition = self.prepare_latents_with_pose(
            image,
            smpl,
            hamer,
            pose,  # Pass pose to the new method
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
        )
        pose_condition = pose_condition.to(transformer_dtype)
        ref_condition = ref_condition.to(transformer_dtype)
        null_ref_condition = null_ref_condition.to(transformer_dtype)

        # 6. TeaCache settings
        if enable_teacache:
            teacache_kwargs = {
                "teacache_thresh": teacache_thresh,
                "accumulated_rel_l1_distance": 0,
                "previous_e0": None,
                "previous_residual": None,
                "use_timestep_proj": use_timestep_proj,
                "coefficients": [
                    8.10705460e+03, 2.13393892e+03, -3.72934672e+02, 1.66203073e+01, -4.17769401e-02
                ] if use_timestep_proj else[
                    -114.36346466, 65.26524496, -18.82220707, 4.91518089, -0.23412683
                ],
                "ret_steps": 5 if use_timestep_proj else 1,
                "cutoff_steps": num_inference_steps
            }
            if self.do_classifier_free_guidance:
                teacache_kwargs_uncond = copy.deepcopy(teacache_kwargs)
        else:
            teacache_kwargs = teacache_kwargs_uncond = None

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                latent_model_input = torch.cat([latents, i2v_condition], dim=1).to(transformer_dtype)
                timestep = t.expand(latents.shape[0])

                noise_pred, teacache_kwargs = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_image=image_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                    add_cond=pose_condition,
                    attn_cond=ref_condition,
                    enable_teacache=enable_teacache,
                    current_step=i,
                    teacache_kwargs=teacache_kwargs,
                )

                if self.do_classifier_free_guidance:
                    noise_uncond, teacache_kwargs_uncond = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=negative_prompt_embeds,
                        encoder_hidden_states_image=null_image_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                        add_cond=pose_condition,
                        attn_cond=null_ref_condition,
                        enable_teacache=enable_teacache,
                        current_step=i,
                        teacache_kwargs=teacache_kwargs_uncond,
                    )
                    noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        if not output_type == "latent":
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = (
                1.0 / torch.tensor(self.vae.config.latents_std)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents = latents / latents_std + latents_mean
            video = self.vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video) 