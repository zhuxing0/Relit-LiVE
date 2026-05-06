from ..models import ModelManager
from ..models.wan_video_dit_relit_live import WanModel
from ..models.wan_video_text_encoder import WanTextEncoder
from ..models.wan_video_vae import WanVideoVAE
from ..models.wan_video_image_encoder import WanImageEncoder
from ..schedulers.flow_match import FlowMatchScheduler
from .base import BasePipeline
from ..prompters import WanPrompter
import torch, os
from einops import rearrange
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Optional
import torch.nn.functional as F

from ..vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear
from ..models.wan_video_text_encoder import T5RelativeEmbedding, T5LayerNorm
from ..models.wan_video_dit_relit_live import RMSNorm, sinusoidal_embedding_1d
from ..models.wan_video_vae import RMS_norm, CausalConv3d, Upsample

class WanVideoRelitlivePipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.float16, tokenizer_path=None):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.model_names = ['text_encoder', 'dit', 'vae']
        self.height_division_factor = 16
        self.width_division_factor = 16

    def enable_vram_management(self, num_persistent_param_in_dit=None):
        dtype = next(iter(self.text_encoder.parameters())).dtype
        enable_vram_management(
            self.text_encoder,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Embedding: AutoWrappedModule,
                T5RelativeEmbedding: AutoWrappedModule,
                T5LayerNorm: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        dtype = next(iter(self.dit.parameters())).dtype
        enable_vram_management(
            self.dit,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv3d: AutoWrappedModule,
                torch.nn.LayerNorm: AutoWrappedModule,
                RMSNorm: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
            max_num_param=num_persistent_param_in_dit,
            overflow_module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        dtype = next(iter(self.vae.parameters())).dtype
        enable_vram_management(
            self.vae,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv2d: AutoWrappedModule,
                RMS_norm: AutoWrappedModule,
                CausalConv3d: AutoWrappedModule,
                Upsample: AutoWrappedModule,
                torch.nn.SiLU: AutoWrappedModule,
                torch.nn.Dropout: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        if self.image_encoder is not None:
            dtype = next(iter(self.image_encoder.parameters())).dtype
            enable_vram_management(
                self.image_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        self.enable_cpu_offload()


    def fetch_models(self, model_manager: ModelManager):
        text_encoder_model_and_path = model_manager.fetch_model("wan_video_text_encoder", require_model_path=True)
        if text_encoder_model_and_path is not None:
            self.text_encoder, tokenizer_path = text_encoder_model_and_path
            self.prompter.fetch_models(self.text_encoder)
            self.prompter.fetch_tokenizer(os.path.join(os.path.dirname(tokenizer_path), "google/umt5-xxl"))
        self.dit = model_manager.fetch_model("wan_video_dit")
        self.vae = model_manager.fetch_model("wan_video_vae")
        self.image_encoder = model_manager.fetch_model("wan_video_image_encoder")


    @staticmethod
    def from_model_manager(model_manager: ModelManager, torch_dtype=None, device=None):
        if device is None: device = model_manager.device
        if torch_dtype is None: torch_dtype = model_manager.torch_dtype
        pipe = WanVideoRelitlivePipeline(device=device, torch_dtype=torch_dtype)
        pipe.fetch_models(model_manager)
        return pipe
    
    
    def denoising_model(self):
        return self.dit


    def encode_prompt(self, prompt, positive=True):
        prompt_emb = self.prompter.encode_prompt(prompt, positive=positive)
        return {"context": prompt_emb}
    
    
    def encode_image(self, image, num_frames, height, width):
        image = self.preprocess_image(image.resize((width, height))).to(self.device)
        clip_context = self.image_encoder.encode_image([image])
        msk = torch.ones(1, num_frames, height//8, width//8, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2)[0]
        
        vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)
        y = self.vae.encode([vae_input.to(dtype=self.torch_dtype, device=self.device)], device=self.device)[0]
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        clip_context = clip_context.to(dtype=self.torch_dtype, device=self.device)
        y = y.to(dtype=self.torch_dtype, device=self.device)
        return {"clip_feature": clip_context, "y": y}


    def tensor2video(self, frames, return_float_array=False):
        frames = rearrange(frames, "C T H W -> T H W C") # (-1, 1)
        if return_float_array:
            frames_float_array = ((frames.float() + 1) * 0.5).clip(0, 1).cpu().numpy().astype(np.float32)
        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        frames = [Image.fromarray(frame) for frame in frames]
        if return_float_array:
            return frames, frames_float_array
        return frames
    
    
    def prepare_extra_input(self, latents=None):
        return {}
    
    
    def encode_video(self, input_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        latents = self.vae.encode(input_video, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return latents
    
    
    def decode_video(self, latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        frames = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return frames


    @torch.no_grad()
    def __call__(
        self,
        prompt,
        negative_prompt="",
        batch=None,
        input_image=None,
        input_video=None,
        denoising_strength=1.0,
        seed=None,
        rand_device="cpu",
        height=480,
        width=832,
        num_frames=81,
        cfg_scale=5.0,
        num_inference_steps=50,
        sigma_shift=5.0,
        tiled=True,
        tile_size=(30, 52),
        tile_stride=(15, 26),
        tea_cache_l1_thresh=None,
        tea_cache_model_id="",
        progress_bar_cmd=tqdm,
        progress_bar_st=None,
        wo_ref_weight=0.0,
        use_muti_ref_image=False,
    ):  

        basecolor = batch["basecolor"]
        depth = batch["depth"]
        metallic = batch["metallic"]
        normal = batch["normal"]
        roughness = batch["roughness"]
        ldr = batch["ldr"]
        hdr_log = batch["hdr_log"]
        env_dir = batch["env_dir"]
        metallic_weight = batch["metallic_weight"]
        roughness_weight = batch["roughness_weight"]
        env_self_weight = batch["env_self_weight"]
        env_cross_weight = batch["env_cross_weight"]

        b = basecolor.shape[0]

        # Parameter check
        height, width = self.check_resize_height_width(height, width)
        if num_frames % 4 != 1:
            num_frames = (num_frames + 2) // 4 * 4 + 1
            print(f"Only `num_frames % 4 != 1` is acceptable. We round it up to {num_frames}.")
        
        # Tiler parameters
        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}

        # Scheduler 
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)

        # Initialize noise
        noise = self.generate_noise((b, 16, ((num_frames - 1) // 4 + 1)*2, height//8, width//8), seed=seed, device=rand_device, dtype=torch.float32) # noise
        noise = noise.to(dtype=self.torch_dtype, device=self.device)
        if input_video is not None:
            self.load_models_to_device(['vae'])
            input_video = self.preprocess_images(input_video)
            input_video = torch.stack(input_video, dim=2).to(dtype=self.torch_dtype, device=self.device)
            latents = self.encode_video(input_video, **tiler_kwargs).to(dtype=self.torch_dtype, device=self.device)
            latents = self.scheduler.add_noise(latents, noise, timestep=self.scheduler.timesteps[0])
        else:
            latents = noise
        
        # Encode Gbuffers (diffusionrenderer)
        self.load_models_to_device(['vae'])
        basecolor = basecolor.to(dtype=self.torch_dtype, device=self.device)
        depth = depth.to(dtype=self.torch_dtype, device=self.device)
        metallic = metallic.to(dtype=self.torch_dtype, device=self.device)
        normal = normal.to(dtype=self.torch_dtype, device=self.device)
        roughness = roughness.to(dtype=self.torch_dtype, device=self.device)

        basecolor_latents = self.encode_video(basecolor, **tiler_kwargs).to(dtype=self.torch_dtype, device=self.device)
        depth_latents = self.encode_video(depth, **tiler_kwargs).to(dtype=self.torch_dtype, device=self.device)
        metallic_latents = self.encode_video(metallic, **tiler_kwargs).to(dtype=self.torch_dtype, device=self.device)
        normal_latents = self.encode_video(normal, **tiler_kwargs).to(dtype=self.torch_dtype, device=self.device)
        roughness_latents = self.encode_video(roughness, **tiler_kwargs).to(dtype=self.torch_dtype, device=self.device)

        # If there are significant deviations in metallic and roughness, we can set their weights to 0 to avoid potential negative effects.
        metallic_weight = metallic_weight.to(dtype=self.torch_dtype, device=self.device)
        roughness_weight = roughness_weight.to(dtype=self.torch_dtype, device=self.device)
        metallic_latents = metallic_latents * metallic_weight.reshape(-1, 1, 1, 1, 1)
        roughness_latents = roughness_latents * roughness_weight.reshape(-1, 1, 1, 1, 1)

        # Process env map (diffusionrenderer)
        ldr = ldr.to(dtype=self.torch_dtype, device=self.device)
        hdr_log = hdr_log.to(dtype=self.torch_dtype, device=self.device)
        env_dir = env_dir.to(dtype=self.torch_dtype, device=self.device)

        h, w = ldr.shape[-2:]
        h_std, w_std = 256, 512
        if (h, w) != (h_std, w_std):
            use_same_env = False
            ldr_cross = ldr.permute(0, 2, 1, 3, 4).contiguous().view(-1, ldr.size(1), ldr.size(3), ldr.size(4)) # b c t h w -> b*t c h w
            ldr_cross = F.interpolate(ldr_cross, size=(h_std, w_std), mode='nearest')
            ldr_cross = ldr_cross.view(ldr.size(0), ldr.size(2), ldr.size(1), h_std, w_std).permute(0, 2, 1, 3, 4) # b*t c h_std w_std -> b c t h_std w_std
            hdr_log_cross = hdr_log.permute(0, 2, 1, 3, 4).contiguous().view(-1, hdr_log.size(1), hdr_log.size(3), hdr_log.size(4))
            hdr_log_cross = F.interpolate(hdr_log_cross, size=(h_std, w_std), mode='nearest')
            hdr_log_cross = hdr_log_cross.view(hdr_log.size(0), hdr_log.size(2), hdr_log.size(1), h_std, w_std).permute(0, 2, 1, 3, 4)
            env_dir_cross = env_dir.permute(0, 2, 1, 3, 4).contiguous().view(-1, env_dir.size(1), env_dir.size(3), env_dir.size(4))
            env_dir_cross = F.interpolate(env_dir_cross, size=(h_std, w_std), mode='nearest')
            env_dir_cross = env_dir_cross.view(env_dir.size(0), env_dir.size(2), env_dir.size(1), h_std, w_std).permute(0, 2, 1, 3, 4)
        else:
            use_same_env = True

        ldr_latents = self.encode_video(ldr, **tiler_kwargs).to(dtype=self.torch_dtype, device=self.device) # (b c f h w) -> (b c_encode f/4 h/8 w/8)
        hdr_log_latents = self.encode_video(hdr_log, **tiler_kwargs).to(dtype=self.torch_dtype, device=self.device)
        env_dir_latents = self.encode_video(env_dir, **tiler_kwargs).to(dtype=self.torch_dtype, device=self.device)
        env_emb = torch.cat([ldr_latents, hdr_log_latents, env_dir_latents], dim=1)
        env_self_weight = env_self_weight.to(dtype=self.torch_dtype, device=self.device)
        env_cross_weight = env_cross_weight.to(dtype=self.torch_dtype, device=self.device)
        env_emb = env_emb * env_self_weight.reshape(-1, 1, 1, 1, 1)

        if not use_same_env:
            ldr_cross = self.encode_video(ldr_cross, **tiler_kwargs).to(dtype=self.torch_dtype, device=self.device)
            hdr_log_cross = self.encode_video(hdr_log_cross, **tiler_kwargs).to(dtype=self.torch_dtype, device=self.device)
            env_dir_cross = self.encode_video(env_dir_cross, **tiler_kwargs).to(dtype=self.torch_dtype, device=self.device)
            env_emb_cross = torch.cat([ldr_cross, hdr_log_cross, env_dir_cross], dim=1)
            env_emb_cross = env_emb_cross * env_cross_weight.reshape(-1, 1, 1, 1, 1)

        if "ref_image" in batch.keys():
            ref_image = batch["ref_image"].to(dtype=self.torch_dtype, device=self.device)
            ref_latent = self.encode_video(ref_image, **tiler_kwargs).to(dtype=self.torch_dtype, device=self.device)
        else:
            ref_image = batch["basecolor"][:,:,0:1,:,:].to(dtype=self.torch_dtype, device=self.device)
            ref_latent = self.encode_video(ref_image, **tiler_kwargs).to(dtype=self.torch_dtype, device=self.device) * 0.0

        # Encode prompts
        self.load_models_to_device(["text_encoder"])
        prompt_emb_posi = self.encode_prompt(prompt, positive=True)
        if cfg_scale != 1.0:
            prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False)
            
        # Encode image
        if input_image is not None and self.image_encoder is not None:
            self.load_models_to_device(["image_encoder", "vae"])
            image_emb = self.encode_image(input_image, num_frames, height, width)
        else:
            image_emb = {}
            
        # Extra input
        extra_input = self.prepare_extra_input(latents)
        
        # TeaCache
        tea_cache_posi = {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id) if tea_cache_l1_thresh is not None else None}
        tea_cache_nega = {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id) if tea_cache_l1_thresh is not None else None}

        # Denoise
        self.load_models_to_device(["dit"])
        tgt_latent_length = latents.shape[2]
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)

            latents_input = torch.cat([latents, basecolor_latents, roughness_latents], dim=2)
            geo_latents = torch.cat([depth_latents, normal_latents], dim=2)
            use_context = True
            if use_same_env:
                other_latents_list = [('latent', metallic_latents)]
            else:
                other_latents_list = [('latent', metallic_latents), ('env', env_emb_cross)]

            if use_muti_ref_image:
                ref_frames = batch["ref_video"].to(dtype=self.torch_dtype, device=self.device) # b c t h w
                ref_index = min(int(progress_id / num_inference_steps * num_frames), num_frames-1)
                ref_image_now = ref_frames[:,:,ref_index:ref_index+1,...]
                ref_latent = self.encode_video(ref_image_now, **tiler_kwargs).to(dtype=self.torch_dtype, device=self.device)

            # Inference
            noise_pred_posi = model_fn_wan_video(self.dit, latents_input, timestep=timestep, geo_latents=geo_latents, env_emb=env_emb, ref_latent=ref_latent, use_context=use_context, other_latents_list=other_latents_list, **prompt_emb_posi, **image_emb, **extra_input, **tea_cache_posi)
            if cfg_scale != 1.0:
                noise_pred_nega = model_fn_wan_video(self.dit, latents_input, timestep=timestep, geo_latents=geo_latents, env_emb=env_emb, ref_latent=ref_latent, use_context=use_context, other_latents_list=other_latents_list, **prompt_emb_nega, **image_emb, **extra_input, **tea_cache_nega)
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            if wo_ref_weight != 0.0 and "ref_image" in batch.keys():
                noise_pred_wo_ref = model_fn_wan_video(self.dit, latents_input, timestep=timestep, geo_latents=geo_latents, env_emb=env_emb, ref_latent=ref_latent*0.0, use_context=use_context, other_latents_list=other_latents_list, **prompt_emb_posi, **image_emb, **extra_input, **tea_cache_posi)
                if wo_ref_weight >= 0:
                    noise_pred = noise_pred / (1+wo_ref_weight) + noise_pred_wo_ref * wo_ref_weight / (1+wo_ref_weight)
                else:
                    noise_pred = noise_pred * (-wo_ref_weight) / (1-wo_ref_weight) + noise_pred_wo_ref / (1-wo_ref_weight)

            # Scheduler
            latents = self.scheduler.step(noise_pred[:,:,:tgt_latent_length,...], self.scheduler.timesteps[progress_id], latents_input[:,:,:tgt_latent_length,...])

        # Decode
        self.load_models_to_device(['vae'])
        frames = self.decode_video(latents[:,:,:(tgt_latent_length//2),...], **tiler_kwargs)
        envs = self.decode_video(latents[:,:,(tgt_latent_length//2):(tgt_latent_length),...], **tiler_kwargs)

        self.load_models_to_device([])
        frames, frames_float_array = self.tensor2video(frames[0], return_float_array=True)
        envs, envs_float_array = self.tensor2video(envs[0], return_float_array=True)

        return frames, envs

class TeaCache:
    def __init__(self, num_inference_steps, rel_l1_thresh, model_id):
        self.num_inference_steps = num_inference_steps
        self.step = 0
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.rel_l1_thresh = rel_l1_thresh
        self.previous_residual = None
        self.previous_hidden_states = None
        
        self.coefficients_dict = {
            "Wan2.1-T2V-1.3B": [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02],
            "Wan2.1-T2V-14B": [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01],
            "Wan2.1-I2V-14B-480P": [2.57151496e+05, -3.54229917e+04,  1.40286849e+03, -1.35890334e+01, 1.32517977e-01],
            "Wan2.1-I2V-14B-720P": [ 8.10705460e+03,  2.13393892e+03, -3.72934672e+02,  1.66203073e+01, -4.17769401e-02],
        }
        if model_id not in self.coefficients_dict:
            supported_model_ids = ", ".join([i for i in self.coefficients_dict])
            raise ValueError(f"{model_id} is not a supported TeaCache model id. Please choose a valid model id in ({supported_model_ids}).")
        self.coefficients = self.coefficients_dict[model_id]

    def check(self, dit: WanModel, x, t_mod):
        modulated_inp = t_mod.clone()
        if self.step == 0 or self.step == self.num_inference_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = self.coefficients
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.step += 1
        if self.step == self.num_inference_steps:
            self.step = 0
        if should_calc:
            self.previous_hidden_states = x.clone()
        return not should_calc

    def store(self, hidden_states):
        self.previous_residual = hidden_states - self.previous_hidden_states
        self.previous_hidden_states = None

    def update(self, hidden_states):
        hidden_states = hidden_states + self.previous_residual
        return hidden_states



def model_fn_wan_video(
    dit: WanModel,
    x: torch.Tensor,
    timestep: torch.Tensor,
    geo_latents: torch.Tensor,
    env_emb: torch.Tensor,
    context: torch.Tensor,
    ref_latent: torch.Tensor,
    other_latents_list: Optional[list] = None,
    use_context: bool = False,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    tea_cache: TeaCache = None,
    **kwargs,
):  

    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep)) # time_embedding
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))

    if use_context:
        context = dit.text_embedding(context) # prompt_embedding
    else:
        context = None
    
    x, (f, h, w) = dit.patchify(x)

    # Reference image
    if len(ref_latent.shape) == 5:
        ref_latent = ref_latent[:, :, 0] # (b c_encode 1 h/8 w/8) -> (b c_encode h/8 w/8)
    ref_latent = dit.ref_conv(ref_latent).flatten(2).transpose(1, 2) # (b c_encode h/8 w/8) -> (b dim h/16? w/16?) -> (b h/16?*w/16? dim)
    x = torch.concat([ref_latent, x], dim=1)
    f += 1

    if geo_latents is not None:
        geo_latents = dit.patch_embedding(geo_latents)
    if env_emb.shape[1] == 16:
        env_emb = dit.patch_embedding(env_emb)
    if other_latents_list is not None:
        other_latents_list = [dit.patch_embedding(latent) if flag == 'latent' else latent for (flag, latent) in other_latents_list]

    f_sample_for_each_latent = (f-1) // 4
    freqs_for_ref = dit.freqs[0][:1]
    freqs_for_latent1 = torch.stack([dit.freqs[0][1+i//5*21+i%5] for i in range(f_sample_for_each_latent)])
    freqs_for_latent2 = torch.stack([dit.freqs[0][6+i//5*21+i%5] for i in range(f_sample_for_each_latent)])
    freqs_for_latent3 = torch.stack([dit.freqs[0][11+i//5*21+i%5] for i in range(f_sample_for_each_latent)])
    freqs_for_latent4 = torch.stack([dit.freqs[0][16+i//5*21+i%5] for i in range(f_sample_for_each_latent)])
    freqs_for_f = torch.concat([freqs_for_ref, freqs_for_latent1, freqs_for_latent2, freqs_for_latent3, freqs_for_latent4], dim=0)
    freqs = torch.cat([
        freqs_for_f.view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device) # 3d ROPE

    # TeaCache
    tea_cache=None
    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, x, t_mod)
    else:
        tea_cache_update = False
    
    if tea_cache_update:
        x = tea_cache.update(x)
    else:
        # blocks
        for block in dit.blocks:
            x = block(x, geo_latents, env_emb, ref_latent, context, other_latents_list, t_mod, freqs)
        if tea_cache is not None:
            tea_cache.store(x)

    x = dit.head(x, t)

    # Remove reference latents
    if ref_latent is not None:
        x = x[:, ref_latent.shape[1]:]
        f -= 1

    x = dit.unpatchify(x, (f, h, w))
    return x
