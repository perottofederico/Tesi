from accelerate.logging import get_logger
import numpy as np
import contextlib
import gc
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from datasets import load_dataset
from diffusers.utils import is_wandb_available
if is_wandb_available():
    import wandb

logger = get_logger(__name__)

def log_validation(
    vae, text_encoder, tokenizer, unet, controlnet, args, accelerator, weight_dtype, step, is_final_validation=False
):
    logger.info("Running validation... ")

    if not is_final_validation:
        controlnet = accelerator.unwrap_model(controlnet)
    else:
        print("Loading  ControlNet model from dir: ", args.output_dir)
        controlnet = ControlNetModel.from_pretrained(args.output_dir, torch_dtype=weight_dtype)

    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    if len(args.validation_image) == len(args.validation_prompt):
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt
    elif len(args.validation_image) == 1:
        validation_images = args.validation_image * len(args.validation_prompt)
        validation_prompts = args.validation_prompt
    elif len(args.validation_prompt) == 1:
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt * len(args.validation_image)
    else:
        raise ValueError(
            "number of `args.validation_image` and `args.validation_prompt` should be checked in `parse_args`"
        )

    image_logs = []
    inference_ctx = contextlib.nullcontext() if is_final_validation else torch.autocast("cuda")
    
    data_val = load_dataset(args.dataset_name, split="validation" if "CVPR" in args.dataset_name else "test",
                            cache_dir=args.cache_dir
                            #cache_dir="/leonardo_scratch/fast/IscrC_GenOpt/luigi/"
                            ).with_format(type='torch')
    if args.subject_num != 0:
        data_val = data_val.filter(lambda x: x['subject'].item() == args.subject_num)

    # for validation_prompt, validation_image in zip(validation_prompts, validation_images):
    for i in range(0, 10):
        #print("Validation step #", i)
        logger.info(f"Validation step {i}")
        # validation_image = Image.open(validation_image).convert("RGB")
        #conditioning image sarebbe
        validation_image = data_val[i]['conditioning_image'].unsqueeze(0).to(accelerator.device) #eeg DEVE essere #,128,512
        #print("\nvalidation_image BEFORE: ", validation_image.shape)
        
        #TODO metto natural image per non condizniore generaizone su label che non ho in inferenza
        #teoricmamente sempre cosi dovrebbe essere in iferenxza
        validation_prompt = args.caption_fixed_string if not args.caption_from_classifier else data_val[i]['caption'] 
        # print(validation_prompt, data_val[i]['label_folder'])
        validation_gt = data_val[i]['image'].unsqueeze(0).to(accelerator.device)
        subjects = data_val[i]['subject'].unsqueeze(0).to(accelerator.device) if "ALL" in args.dataset_name else torch.tensor([4]).unsqueeze(0).to(accelerator.device)
        validation_image = controlnet.controlnet_cond_embedding(validation_image, subjects, return_vector = False)
        #print("\nvalidation_image AFTER: ", validation_image.shape)
        images = []

        for _ in range(args.num_validation_images):
            with inference_ctx:
                image = pipeline(
                    validation_prompt, validation_image, num_inference_steps=20, generator=generator, subjects = subjects
                ).images[0]

            images.append(image)

        validation_prompt = data_val[i]['caption'] # i want logged the class name even if i choose to use the fixed caption
        image_logs.append(
            {"validation_image": validation_gt, "images": images, "validation_prompt": validation_prompt}
        )

    tracker_key = "test" if is_final_validation else "validation"
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]

                formatted_images = []

                formatted_images.append(np.asarray(validation_image))

                for image in images:
                    formatted_images.append(np.asarray(image))

                formatted_images = np.stack(formatted_images)

                tracker.writer.add_images(validation_prompt, formatted_images, step, dataformats="NHWC")
        elif tracker.name == "wandb":
            formatted_images = []

            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]

                formatted_images.append(wandb.Image(validation_image, caption="Ground truth"))

                for image in images:
                    image = wandb.Image(image, caption=validation_prompt)
                    formatted_images.append(image)

            tracker.log({tracker_key: formatted_images})
        else:
            logger.warning(f"image logging not implemented for {tracker.name}")

        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

        return image_logs