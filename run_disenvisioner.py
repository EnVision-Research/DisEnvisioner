import os
import argparse
import logging
from PIL import Image
from contextlib import nullcontext

import torch
from diffusers import UNet2DConditionModel
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor, CLIPTextModelWithProjection
from diffusers import StableDiffusionPipeline

from disvisioner_modules.disvisioner import DisVisioner
from envisioner_modules.envisioner import Projector, EnVisioner
from envisioner_modules.attention_processor import EVAttnProcessor2_0 as EVAttnProcessor, AttnProcessor2_0 as AttnProcessor
from utils import seed_all, is_torch2_available, image_grid
assert is_torch2_available()


def set_scales(pipe, scale_object, scale_others):
    logging.info(f"==> setting scales: scale_obj {scale_object}, scale_others {scale_others}")
    for attn_processor in pipe.unet.attn_processors.values():
        if isinstance(attn_processor, EVAttnProcessor):
            attn_processor.scale_object = scale_object
            attn_processor.scale_others = scale_others

@torch.inference_mode()
def run_inference(args, unet, disv_image_encoder, disv_text_encoder, disvisioner, envisioner, device, dtype):    
    # load SD pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=unet,
        torch_dtype=dtype,
    ).to(device)

    # set scales
    set_scales(pipe, args.scale_object, args.scale_others)

    infer_image = args.infer_image
    infer_prompt = args.infer_prompt
    negative_prompt = args.negative_prompt
    print(f"Running for image: {infer_image} with prompt: {infer_prompt}")

    # ------------------ prepare textual embedding ------------------
    class_name = args.class_name
    infer_prompt = infer_prompt.replace("*", class_name)
    print(infer_prompt)

    prompt_embeds_, negative_prompt_embeds_ = pipe.encode_prompt(
        infer_prompt,
        device=device,
        num_images_per_prompt=args.num_samples,
        do_classifier_free_guidance=True,
        negative_prompt=negative_prompt,
    )

    # ------------------ prepare image embedding using DisEnvisioner ------------------
    # read image prompt
    image = Image.open(infer_image)
    image = image.resize((256, 256))
    # get_image_embeds
    clip_image = CLIPImageProcessor()(images=[image], return_tensors="pt").pixel_values
    clip_image = clip_image.to(device, dtype=dtype)
    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(pipe.device.type)
    with autocast_ctx:
        # disvisioner
        image_features = disv_image_encoder(clip_image.to(device, dtype=dtype), output_hidden_states=True)
        image_embeddings = image_features.last_hidden_state
        image_embeddings = image_embeddings.detach()
        
        class_ids = pipe.tokenizer(
            class_name,
            padding="max_length",
            truncation=True,
            max_length=pipe.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0].to(device).unsqueeze(0)
        
        class_proj = disv_text_encoder(class_ids.to(device)).text_embeds
        inj_embedding = disvisioner(image_embeddings, class_proj)

        # envisioner
        # get projected embeds
        image_prompt_embeds_object, image_prompt_embeds_others = envisioner.image_proj_model(inj_embedding.float())            
        uncond_image_prompt_embeds_object, uncond_image_prompt_embeds_others = envisioner.image_proj_model(torch.zeros_like(inj_embedding.float()))

        bs_embed, seq_len, _ = image_prompt_embeds_object.shape
        image_prompt_embeds_object = image_prompt_embeds_object.repeat(1, args.num_samples, 1).view(bs_embed * args.num_samples, seq_len, -1)
        uncond_image_prompt_embeds_object = uncond_image_prompt_embeds_object.repeat(1, args.num_samples, 1).view(bs_embed * args.num_samples, seq_len, -1)
        
        bs_embed, seq_len, _ = image_prompt_embeds_others.shape
        image_prompt_embeds_others = image_prompt_embeds_others.repeat(1, args.num_samples, 1).view(bs_embed * args.num_samples, seq_len, -1)
        uncond_image_prompt_embeds_others = uncond_image_prompt_embeds_others.repeat(1, args.num_samples, 1).view(bs_embed * args.num_samples, seq_len, -1)
        
        # ------------------ prepare textual and image embeddings ------------------
        prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds_object, image_prompt_embeds_others], dim=1)
        negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds_object, uncond_image_prompt_embeds_others], dim=1)

        # gen
        generator = torch.Generator(device).manual_seed(args.seed) if args.seed is not None else None
        images = pipe(
            prompt_embeds=prompt_embeds,
            height=args.resolution,
            width=args.resolution,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            generator=generator
        ).images

    # save
    gen_images = [image.resize((args.resolution, args.resolution))] + images
    grid = image_grid(gen_images, len(gen_images)//(1+args.num_samples), 1+args.num_samples)
    # val_file_name = '.'.join(os.path.basename(infer_image).split(".")[:-1])
    grid.save(os.path.join(args.output_dir, f'sobj{args.scale_object}_soth{args.scale_others}_[{infer_prompt}]_seed{args.seed}.png'))

    
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--half_precision",
        action="store_true"
    )
    parser.add_argument(
        "--disvisioner_path",
        type=str,
        default=None,
        help="Path to pretrained disvisioner.",
        required=True
    )
    parser.add_argument(
        "--token_num",
        type=int,
        default=1,
        help="number of tokens for object"
    )
    parser.add_argument(
        "--pretrained_CLIP",
        type=str,
        default=None,
        help="Path to pretrained disvisioner encoders.",
        required=True
    )
    parser.add_argument(
        "--scale_object",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--scale_others",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="disenvisioner",
        help="The output directory where the model predictions will be written.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images"
        ),
    )
    parser.add_argument(
        "--infer_image",
        type=str,
        required=True
    )
    parser.add_argument(
        "--infer_prompt",
        type=str,
        required=True
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="monochrome, lowres, bad anatomy, worst quality, low quality"
    )
    parser.add_argument(
        "--class_name",
        type=str,
        required=True
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=4
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50
    )
    parser.add_argument(
        "--envisioner_path",
        type=str,
        default=None,
        help="Path to pretrained envisioner.",
    )
    parser.add_argument(
        "--object_factor",
        type=int,
        default=1,
        help="factor that determines the number of object tokens. "
    )
    parser.add_argument(
        "--others_factor",
        type=int,
        default=1,
        help="factor that determines the number of other component tokens."
    ) 
    args = parser.parse_args()

    return args
    
def main():
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Run inference...")
    
    args = parse_args()
    # ------------------ Preparation ------------------
    os.makedirs(args.output_dir, exist_ok=True)
    logging.info(f"Output dir = {args.output_dir}")

    if args.seed is not None:
        seed_all(args.seed)

    # half_precision
    if args.half_precision:
        dtype = torch.float16
        logging.info(f"Running with half precision ({dtype}).")
    else:
        dtype = torch.float32

    # -------------------- Device --------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"Device = {device}")

    # -------------------- Models --------------------
    # ------------------ 1. Load Unet model from SD ------------------
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    
    # ------------------ 2. Load models for disvisioner ------------------
    disv_text_encoder = CLIPTextModelWithProjection.from_pretrained(args.pretrained_CLIP)
    disv_image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.pretrained_CLIP)
    # define disvisioner
    disvisioner = DisVisioner(
        image_hidden_size=disv_image_encoder.vision_model.config.hidden_size, 
        text_hidden_size=disv_text_encoder.text_model.config.hidden_size,
        output_dim=disv_text_encoder.text_model.config.hidden_size,
        token_num=args.token_num, num_refine=2
        )
    logging.info(f"Load Disvisioner from {args.disvisioner_path}")
    disvisioner.load_state_dict(torch.load(args.disvisioner_path, map_location='cpu'), strict=True)

    # ------------------ 3. Load envisioner ------------------
    # 1. define projector
    image_projector = Projector(
        cross_attention_dim=unet.config.cross_attention_dim, # output dim
        input_embedding_dim=disv_text_encoder.text_model.config.hidden_size, # input dim
        clip_extra_context_tokens=4,
    )
    # 2. Define additional CA modules (referred to IP-Adapter)
    attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        
        if cross_attention_dim is None: # attn1
            attn_procs[name] = AttnProcessor()
        else: # attn2
            attn_procs[name] = EVAttnProcessor(hidden_size=hidden_size,                                                cross_attention_dim=cross_attention_dim, 
                                               num_tokens_object=args.object_factor*4, # number of object tokens
                                               num_tokens_others=args.others_factor*4, # number of other tokens
                                               )
    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())

    # 3. Define envisioner (Projector + added adapter_modules)
    envisioner = EnVisioner(image_projector, adapter_modules)
    
    logging.info(f"Load EnVisioner from {args.envisioner_path}")

    envisioner.load_state_dict(torch.load(args.envisioner_path),strict=False)

    unet.to(device, dtype=dtype)
    disv_image_encoder.to(device, dtype=dtype)
    disv_text_encoder.to(device, dtype=dtype)
    disvisioner.to(device, dtype=dtype)
    envisioner.to(device, dtype=dtype)

    run_inference(args, unet, disv_image_encoder, disv_text_encoder, disvisioner, envisioner, device, dtype)


if __name__ == "__main__":
    main()    
