import json
import os
import re
import shutil
import subprocess
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import cv2
import random
import numpy as np
import torch
from cog import BasePredictor, Input, Path
from diffusers import (
    DDIMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    UniPCMultistepScheduler,
)
from diffusers.models.attention_processor import LoRAAttnProcessor2_0
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from diffusers.utils import load_image
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import CLIPImageProcessor

import json
import requests
from io import BytesIO
import tarfile
import torch
from PIL import Image
import shutil
import math
import cog

from dataset_and_utils import TokenEmbeddingsHandler

SDXL_MODEL_CACHE = "./sdxl-cache"
CONTROLNET_CACHE = "./controlnet_cache"
SAFETY_CACHE = "./safety-cache"
SDXL_URL = "https://weights.replicate.delivery/default/sdxl/sdxl-vae-fix-1.0.tar"
REFINER_URL = (
    "https://weights.replicate.delivery/default/sdxl/refiner-no-vae-no-encoder-1.0.tar"
)
SAFETY_URL = "https://weights.replicate.delivery/default/sdxl/safety-1.0.tar"


class KarrasDPM:
    def from_config(config):
        return DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True)


SCHEDULERS = {
    "DDIM": DDIMScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "HeunDiscrete": HeunDiscreteScheduler,
    "KarrasDPM": KarrasDPM,
    "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler,
    "K_EULER": EulerDiscreteScheduler,
    "PNDM": PNDMScheduler,
    "UNIPC": UniPCMultistepScheduler,
}


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_output(["pget", "-x", url, dest])
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def load_trained_weights(self, weights_url, pipe, use_default=True):
        if use_default:
            self.is_lora = True
            lora_default_path = "./lora/default_lora.safetensors"
            print("Loading Unet with default LoRA")

            unet = pipe.unet

            tensors = load_file(lora_default_path)

            unet = pipe.unet
            unet_lora_attn_procs = {}
            name_rank_map = {}
            for tk, tv in tensors.items():
                # up is N, d
                if tk.endswith("up.weight"):
                    proc_name = ".".join(tk.split(".")[:-3])
                    r = tv.shape[1]
                    name_rank_map[proc_name] = r

            for name, attn_processor in unet.attn_processors.items():
                cross_attention_dim = (
                    None
                    if name.endswith("attn1.processor")
                    else unet.config.cross_attention_dim
                )
                if name.startswith("mid_block"):
                    hidden_size = unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(unet.config.block_out_channels))[
                        block_id
                    ]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = unet.config.block_out_channels[block_id]

                module = LoRAAttnProcessor2_0(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    rank=name_rank_map[name],
                )
                unet_lora_attn_procs[name] = module.to("cuda")

            unet.set_attn_processor(unet_lora_attn_procs)
            unet.load_state_dict(tensors, strict=False)
            return

        # Get the TAR archive content
        weights_tar_data = requests.get(weights_url).content

        with tarfile.open(fileobj=BytesIO(weights_tar_data), mode="r") as tar_ref:
            tar_ref.extractall("trained-model")

        local_weights_cache = "./trained-model"

        # load UNET
        print("Loading fine-tuned model")
        self.is_lora = False

        maybe_unet_path = os.path.join(local_weights_cache, "unet.safetensors")
        if not os.path.exists(maybe_unet_path):
            print("Does not have Unet. Assume we are using LoRA")
            self.is_lora = True

        if not self.is_lora:
            print("Loading Unet")

            new_unet_params = load_file(
                os.path.join(local_weights_cache, "unet.safetensors")
            )
            sd = pipe.unet.state_dict()
            sd.update(new_unet_params)
            pipe.unet.load_state_dict(sd)

        else:
            print("Loading Unet LoRA")

            unet = pipe.unet

            tensors = load_file(os.path.join(local_weights_cache, "lora.safetensors"))

            unet = pipe.unet
            unet_lora_attn_procs = {}
            name_rank_map = {}
            for tk, tv in tensors.items():
                # up is N, d
                if tk.endswith("up.weight"):
                    proc_name = ".".join(tk.split(".")[:-3])
                    r = tv.shape[1]
                    name_rank_map[proc_name] = r

            for name, attn_processor in unet.attn_processors.items():
                cross_attention_dim = (
                    None
                    if name.endswith("attn1.processor")
                    else unet.config.cross_attention_dim
                )
                if name.startswith("mid_block"):
                    hidden_size = unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(unet.config.block_out_channels))[
                        block_id
                    ]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = unet.config.block_out_channels[block_id]

                module = LoRAAttnProcessor2_0(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    rank=name_rank_map[name],
                )
                unet_lora_attn_procs[name] = module.to("cuda")

            unet.set_attn_processor(unet_lora_attn_procs)
            unet.load_state_dict(tensors, strict=False)

        # load text
        handler = TokenEmbeddingsHandler(
            [pipe.text_encoder, pipe.text_encoder_2], [pipe.tokenizer, pipe.tokenizer_2]
        )
        if os.path.exists(os.path.join(local_weights_cache, "embeddings.pti")):
            handler.load_embeddings(os.path.join(local_weights_cache, "embeddings.pti"))

            with open(
                os.path.join(local_weights_cache, "special_params.json"), "r"
            ) as f:
                params = json.load(f)
            self.token_map = params

            self.tuned_model = True

    def setup(self, weights: Optional[Path] = None):
        """Load the model into memory to make running multiple predictions efficient"""
        start = time.time()
        self.tuned_model = False

        print("Loading safety checker...")
        if not os.path.exists(SAFETY_CACHE):
            download_weights(SAFETY_URL, SAFETY_CACHE)

        if not os.path.exists(SDXL_MODEL_CACHE):
            download_weights(SDXL_URL, SDXL_MODEL_CACHE)

        controlnet = ControlNetModel.from_pretrained(
            CONTROLNET_CACHE,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
        self.txt2img_pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "./sdxl-cache",
            controlnet=controlnet,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
        self.txt2img_pipe.load_lora_weights("./lora/default_lora.safetensors")
        self.txt2img_pipe.fuse_lora(lora_scale=0.7)
        self.txt2img_pipe.load_lora_weights("./lora/default_text.safetensors")
        self.txt2img_pipe.fuse_lora(lora_scale=0.7)
        self.txt2img_pipe.to("cuda")
        self.txt2img_pipe.enable_xformers_memory_efficient_attention()
        self.is_lora = True

        print("setup took: ", time.time() - start)
        # self.txt2img_pipe.__class__.encode_prompt = new_encode_prompt

    def load_image(self, path):
        # Copy the image to a temporary location
        shutil.copyfile(path, "/tmp/image.png")

        # Open the copied image
        img = Image.open("/tmp/image.png")

        # Calculate the new dimensions while maintaining aspect ratio
        width, height = img.size
        new_width = math.ceil(width / 64) * 64
        new_height = math.ceil(height / 64) * 64

        # Resize the image if needed
        if new_width != width or new_height != height:
            img = img.resize((new_width, new_height))

        # Convert the image to RGB mode
        img = img.convert("RGB")

        return img

    @torch.inference_mode()
    def predict(
        self,
        prompt1: str = Input(
            description="Input prompt",
            default=None,
        ),
        prompt2: str = Input(
            description="Input prompt",
            default=None,
        ),
        prompt3: str = Input(
            description="Input prompt",
            default=None,
        ),
        prompt4: str = Input(
            description="Input prompt",
            default=None,
        ),
        negative_prompt: str = Input(
            description="Input Negative Prompt",
            default="",
        ),
        image1: Path = Input(
            description="Input image for img2img or inpaint mode",
            default=None,
        ),
        image2: Path = Input(
            description="Input image for img2img or inpaint mode",
            default=None,
        ),
        image3: Path = Input(
            description="Input image for img2img or inpaint mode",
            default=None,
        ),
        image4: Path = Input(
            description="Input image for img2img or inpaint mode",
            default=None,
        ),
        width1: int = Input(
            description="Width of output image",
            default=1024,
        ),
        height1: int = Input(
            description="Height of output image",
            default=1024,
        ),
        height2: int = Input(
            description="Height of output image",
            default=1024,
        ),
        width2: int = Input(
            description="Width of output image",
            default=1024,
        ),
        height3: int = Input(
            description="Height of output image",
            default=1024,
        ),
        width3: int = Input(
            description="Width of output image",
            default=1024,
        ),
        height4: int = Input(
            description="Height of output image",
            default=1024,
        ),
        width4: int = Input(
            description="Width of output image",
            default=1024,
        ),
        scheduler: str = Input(
            description="scheduler",
            choices=SCHEDULERS.keys(),
            default="K_EULER",
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=50, default=7.5
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        controlnet_conditioning_scale: float = Input(
            description="ControlNet conditioning scale. Only applicable on trained models.",
            ge=0.0,
            le=1.0,
            default=0.6,
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""

        print(f"Using seed: {seed}")

        sdxl_kwargs = {}
        self.tuned_model = False
        images_list = []
        for image in [image1, image2, image3, image4]:
            if image:
                images_list.append(image)
        images_input = []
        height_list = [height1, height2, height3, height4]
        width_list = [width1, width2, width3, width4]
        for idx, image in enumerate(images_list):
            # if condition to check if it is pil image or not
            print(image)
            print(type(image))
            if isinstance(image, str) or isinstance(image, cog.types.Path):
                height = height_list[idx]
                width = width_list[idx]
                image_ = self.load_image(image)
                image_ = np.array(image_.resize((width, height)))
                LOW_THRES = 100
                HIGH_THRES = 200
                image_ = cv2.Canny(image_, LOW_THRES, HIGH_THRES)
                image_ = image_[:, :, None]
                image_ = np.concatenate([image_, image_, image_], axis=2)
                image_ = Image.fromarray(image_)
                images_input.append(image_)

        # if len(images_input) == 1:
        #     images_input = images_input[0]
        sdxl_kwargs["image"] = images_input
        prompt_list = []
        generators = []
        for prompt in [prompt1, prompt2, prompt3, prompt4]:
            if prompt:
                prompt_list.append(prompt)
        if len(images_input) > len(prompt_list):
            images_input = images_input[: len(prompt_list)]
        if len(images_input) < len(prompt_list):
            # duplicate the images
            images_input = images_input * (len(prompt_list) // len(images_input))
            images_input = (
                images_input + images_input[: len(prompt_list) % len(images_input)]
            )

        negative_prompt_list = [negative_prompt] * len(prompt_list)

        # negative_prompt_list = [negative_prompt] * len(prompt_list)

        if seed is None:
            seed = [random.randint(0, 9999999999) for _ in prompt_list]
            for s in seed:
                generators.append(torch.Generator("cuda").manual_seed(s))
        else:
            seed = [seed] * len(prompt_list)
            for s in seed:
                generators.append(torch.Generator("cuda").manual_seed(s))

        pipe = self.txt2img_pipe
        pipe.scheduler = SCHEDULERS[scheduler].from_config(pipe.scheduler.config)
        output_images = []
        for idx, prompt in enumerate(prompt_list):
            sdxl_kwargs["image"] = images_input[idx]
            sdxl_kwargs["width"] = width_list[idx]
            sdxl_kwargs["height"] = height_list[idx]
            sdxl_kwargs["controlnet_conditioning_scale"] = controlnet_conditioning_scale

            common_args = {
                "prompt": prompt,
                "negative_prompt": negative_prompt_list[idx],
                "guidance_scale": guidance_scale,
                "generator": generators[idx],
                "num_inference_steps": num_inference_steps,
            }
            sdxl_kwargs["cross_attention_kwargs"] = {"scale": 1.0}
            output = pipe(**common_args, **sdxl_kwargs)
            output_images.append(output.images[0])

        has_nsfw_content = [False] * len(output_images)

        output_paths = []
        for i, nsfw in enumerate(has_nsfw_content):
            if nsfw:
                print(f"NSFW content detected in image {i}")
                continue
            output_path = f"/tmp/out-{i}.png"
            output_images[i].save(output_path)
            output_paths.append(Path(output_path))

        if len(output_paths) == 0:
            raise Exception(
                f"NSFW content detected. Try running it again, or try a different prompt."
            )

        return output_paths
