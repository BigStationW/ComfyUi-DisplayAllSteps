import torch
import comfy.sample
import comfy.utils
import comfy.model_management
import latent_preview


class SamplerCustomAdvancedAllSteps:
    """Sampler that outputs latents from all intermediate steps."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise": ("NOISE",),
                "guider": ("GUIDER",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "latent_image": ("LATENT",),
            }
        }
    
    RETURN_TYPES = ("LATENT", "LATENT", "LATENT")
    RETURN_NAMES = ("output", "denoised_output", "all_steps_latents")
    FUNCTION = "sample"
    CATEGORY = "sampling/custom_sampling"

    def sample(self, noise, guider, sampler, sigmas, latent_image):
        latent = latent_image
        latent_image = latent["samples"]
        latent = latent.copy()
        latent_image = comfy.sample.fix_empty_latent_channels(guider.model_patcher, latent_image)
        latent["samples"] = latent_image

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        x0_output = {}
        
        # Storage for all intermediate DENOISED latents (x0, not x)
        all_step_latents = []
        
        original_callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1, x0_output)
        
        def capture_callback(step, x0, x, total_steps):
            # Store x0 (predicted clean image), NOT x (noisy latent)
            if x0 is not None:
                # Process through model's latent output processing
                processed_x0 = guider.model_patcher.model.process_latent_out(x0.cpu())
                all_step_latents.append(processed_x0.clone())
            
            if original_callback is not None:
                original_callback(step, x0, x, total_steps)
        
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        samples = guider.sample(
            noise.generate_noise(latent), 
            latent_image, 
            sampler, 
            sigmas, 
            denoise_mask=noise_mask, 
            callback=capture_callback, 
            disable_pbar=disable_pbar, 
            seed=noise.seed
        )
        samples = samples.to(comfy.model_management.intermediate_device())

        out = latent.copy()
        out["samples"] = samples
        
        if "x0" in x0_output:
            out_denoised = latent.copy()
            out_denoised["samples"] = guider.model_patcher.model.process_latent_out(x0_output["x0"].cpu())
        else:
            out_denoised = out
        
        # Create output containing all step latents as a batch
        all_steps_out = latent.copy()
        if all_step_latents:
            stacked_latents = torch.cat(all_step_latents, dim=0)
            all_steps_out["samples"] = stacked_latents
            all_steps_out["batch_index"] = list(range(len(all_step_latents)))
        else:
            all_steps_out["samples"] = samples
            
        return (out, out_denoised, all_steps_out)


class VAEDecodeAllSteps:
    """Decode a batch of latents (from all steps) into images."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "decode"
    CATEGORY = "latent"

    def decode(self, samples, vae):
        latents = samples["samples"]
        batch_size = latents.shape[0]
        
        decoded_images = []
        
        for i in range(batch_size):
            single_latent = latents[i:i+1]
            decoded = vae.decode(single_latent)
            decoded_images.append(decoded)
        
        all_images = torch.cat(decoded_images, dim=0)
        
        return (all_images,)


NODE_CLASS_MAPPINGS = {
    "SamplerCustomAdvancedAllSteps": SamplerCustomAdvancedAllSteps,
    "VAEDecodeAllSteps": VAEDecodeAllSteps,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SamplerCustomAdvancedAllSteps": "Sampler (Custom Advanced All Steps)",
    "VAEDecodeAllSteps": "VAE Decode All Steps",
}