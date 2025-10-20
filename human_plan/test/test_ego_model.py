import transformers
import torch
from llava.train.train import ModelArguments, DataArguments, TrainingArguments
from llava.model import *
from human_plan.model import *
from human_plan.dataset import *

import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from PIL import Image

import requests
from PIL import Image
from io import BytesIO

from human_plan.model import *
from human_plan.model.ego_model import EgoManipLLAMAConfig, EgoManipModelForRegression


def load_image(image_file):
  if image_file.startswith('http://') or image_file.startswith('https://'):
    response = requests.get(image_file)
    image = Image.open(BytesIO(response.content)).convert('RGB')
  else:
    image = Image.open(image_file).convert('RGB')
  return image


class VisionTowerConfig:
  def __init__(self,):
    super().__init__()
    self.vision_tower = "openai/clip-vit-large-patch14-336"
    print(self.vision_tower)
    self.mm_vision_select_layer = -1
    self.mm_vision_select_feature = "patch"
    self.version = "v0"
    self.freeze_backbone = False
    self.tune_mm_mlp_adapter = False
    # self.vision_tower=None
    self.mm_vision_select_layer = -1   # default to the last layer
    self.pretrain_mm_mlp_adapter = None
    self.mm_projector_type = 'linear'
    self.mm_use_im_start_end = False
    self.mm_use_im_patch_token = True
    self.mm_patch_merge_type = 'flat'
    self.mm_vision_select_feature = "patch"


class VLMConfig:
  def __init__(self,):
    super().__init__()
    self.vlm_path = "liuhaotian/llava-v1.5-7b"

    self.vlm_select_layer = -1
    self.vlm_select_feature = "patch"
    self.load_8bit = False
    self.load_4bit = True
    self.pretrain_vlm_mlp_adapter = None
    # self.version="v0"
    # self.freeze_backbone=False
    # self.tune_mm_mlp_adapter=False
    # # self.vision_tower=None
    # self.mm_vision_select_layer=-1   # default to the last layer
    self.vlm_base = None
    self.mm_projector_type = 'linear'
    self.mm_use_im_start_end = False
    self.mm_use_im_patch_token = True
    self.mm_patch_merge_type = 'flat'
    self.mm_vision_select_feature = "patch"

    self.max_new_tokens = 512
    self.temperature = 0.2


def main(args):
  # Model
  # disable_torch_init()
  model_name = get_model_name_from_path(args.model_path)
  print("-------------")
  print(args.model_path)
  print(args.model_base)
  print(model_name)
  print("-------------")
  # tokenizer, model, image_processor, context_len = load_pretrained_model(
  #     args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device
  # )

  ego_config = EgoManipLLAMAConfig(
      vision_tower="facebook/dinov2-base",
      vlm_base="liuhaotian/llava-v1.5-7b",
      intermediate_size=4096,
      num_hidden_layers=10
  )

  ego_model = EgoManipModelForRegression(
      config=ego_config
  )
  # ego_model.build_vlm()
  vision_config = VisionTowerConfig()
  # print(vision_config)
  # print(vision_config.vision_tower)
  ego_model.get_model().initialize_vision_modules(ego_config)

  # exit()
  vlm_config = VLMConfig()
  ego_model.get_model().initialize_vlm_modules(ego_config)

  ego_model = ego_model.to(args.device).half()
  # .to(args.device).bfloat16()
  print(ego_model)
  # exit()

  image1 = load_image(args.image_file)
  image2 = load_image(args.image_file)
  image3 = load_image(args.image_file)

  # Similar operation in model_worker.py
  vision_image_tensor = process_images(
      [image1, image2, image3], ego_model.vision_image_processor(), ego_model.config
  )
  # vlm_image_tensor = process_images(
  #   [image1, image2], ego_model.vlm_image_processor(), ego_model.config
  # )
  if type(vision_image_tensor) is list:
    vision_image_tensor = [image.to(ego_model.device, dtype=torch.float16)
                           for image in vision_image_tensor]
    # vlm_image_tensor = [image.to(ego_model.device, dtype=torch.float16)
    #                 for image in vlm_image_tensor]
  else:
    vision_image_tensor = vision_image_tensor.to(
        ego_model.device, dtype=torch.float16)

  print(vision_image_tensor)
  print(vision_image_tensor.shape)
  # exit()
  # vlm_image_tensor = vlm_image_tensor.to(ego_model.device, dtype=torch.float16)

  inp1 = "What's in the image"
  inp2 = "What's in the image, tell me in more detail and tell me what a dog would do in this scenario"
  inp3 = "What's in the image, tell me in more detail and tell me what a dog would do in this scenario. What if someone need to do some manipulation task here?"

  text_inputs = [
      inp1, inp2, inp3
  ]

  # conv.append_message(conv.roles[0], inp)
  # conv.append_message(conv.roles[1], None)
  # prompt = conv.get_prompt()

  vlm_input_ids, vlm_image_tensors, image_sizes, attention_mask = ego_model.get_vlm().vlm_preprocessing(
      [image1, image2, image3], text_inputs
  )

  print(ego_model.config)
  hand_inputs = torch.rand(
      3, ego_model.config.hand_input_dim
  ).to(ego_model.device).half()
  with torch.inference_mode():
    output = ego_model.forward(
        vlm_input_ids=vlm_input_ids,
        vlm_attention_mask=attention_mask,
        vision_images=vision_image_tensor,
        vlm_images=vlm_image_tensors,
        image_sizes=image_sizes,
        # image_sizes=[image_size],
        # text_inputs = text_inputs,
        # do_sample=True if args.temperature > 0 else False,
        # do_sample=False,
        # temperature=args.temperature,
        # max_new_tokens=args.max_new_tokens,
        # streamer=streamer,
        hand_inputs=hand_inputs,
        output_hidden_states=True,
        return_dict=True,
        # use_cache=True
    )
    print(output.keys())
    # print(len(output["hidden_states"]))
    # for hidden_state in output_ids["hidden_states"]:
    #   print(len(hidden_state))
    #   for item in hidden_state:
    #     print(item.shape)
    # all_hidden_state
  # print(output)
  # outputs = tokenizer.decode(output_ids["sequences"][0]).strip()
  # conv.messages[-1][-1] = outputs

  if args.debug:
    print("\n", {"prompt": prompt, "outputs": outputs}, "\n")

# def train(attn_implementation=None):
#     global local_rank

#     parser = transformers.HfArgumentParser(
#         (ModelArguments, DataArguments, TrainingArguments))
#     model_args, data_args, training_args = parser.parse_args_into_dataclasses()
#     local_rank = training_args.local_rank
#     compute_dtype = (torch.float16 if training_args.fp16 else (
#         torch.bfloat16 if training_args.bf16 else torch.float32))

#     bnb_model_from_pretrained_args = {}
#     if training_args.bits in [4, 8]:
#         from transformers import BitsAndBytesConfig
#         bnb_model_from_pretrained_args.update(dict(
#             device_map={"": training_args.device},
#             load_in_4bit=training_args.bits == 4,
#             load_in_8bit=training_args.bits == 8,
#             quantization_config=BitsAndBytesConfig(
#                 load_in_4bit=training_args.bits == 4,
#                 load_in_8bit=training_args.bits == 8,
#                 llm_int8_skip_modules=["mm_projector"],
#                 llm_int8_threshold=6.0,
#                 llm_int8_has_fp16_weight=False,
#                 bnb_4bit_compute_dtype=compute_dtype,
#                 bnb_4bit_use_double_quant=training_args.double_quant,
#                 bnb_4bit_quant_type=training_args.quant_type  # {'fp4', 'nf4'}
#             )
#         ))

#     if model_args.vision_tower is not None:
#         if 'mpt' in model_args.model_name_or_path:
#             config = transformers.AutoConfig.from_pretrained(
#                 model_args.model_name_or_path, trust_remote_code=True)
#             config.attn_config['attn_impl'] = training_args.mpt_attn_impl
#             model = LlavaMptForCausalLM.from_pretrained(
#                 model_args.model_name_or_path,
#                 config=config,
#                 cache_dir=training_args.cache_dir,
#                 **bnb_model_from_pretrained_args
#             )
#         else:
#             model = LlavaLlamaForCausalLM.from_pretrained(
#                 model_args.model_name_or_path,
#                 cache_dir=training_args.cache_dir,
#                 attn_implementation=attn_implementation,
#                 torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
#                 **bnb_model_from_pretrained_args
#             )
#     else:
#         model = transformers.LlamaForCausalLM.from_pretrained(
#             model_args.model_name_or_path,
#             cache_dir=training_args.cache_dir,
#             attn_implementation=attn_implementation,
#             torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
#             **bnb_model_from_pretrained_args
#         )
#     model.config.use_cache = False

#     if model_args.freeze_backbone:
#         model.model.requires_grad_(False)

#     if training_args.bits in [4, 8]:
#         from peft import prepare_model_for_kbit_training
#         model.config.torch_dtype = (torch.float32 if training_args.fp16 else (
#             torch.bfloat16 if training_args.bf16 else torch.float32))
#         model = prepare_model_for_kbit_training(
#             model, use_gradient_checkpointing=training_args.gradient_checkpointing)

#     if training_args.gradient_checkpointing:
#         if hasattr(model, "enable_input_require_grads"):
#             model.enable_input_require_grads()
#         else:
#             def make_inputs_require_grad(module, input, output):
#                 output.requires_grad_(True)
#             model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

#     if training_args.lora_enable:
#         from peft import LoraConfig, get_peft_model
#         lora_config = LoraConfig(
#             r=training_args.lora_r,
#             lora_alpha=training_args.lora_alpha,
#             target_modules=find_all_linear_names(model),
#             lora_dropout=training_args.lora_dropout,
#             bias=training_args.lora_bias,
#             task_type="CAUSAL_LM",
#         )
#         if training_args.bits == 16:
#             if training_args.bf16:
#                 model.to(torch.bfloat16)
#             if training_args.fp16:
#                 model.to(torch.float16)
#         rank0_print("Adding LoRA adapters...")
#         model = get_peft_model(model, lora_config)

#     if 'mpt' in model_args.model_name_or_path:
#         tokenizer = transformers.AutoTokenizer.from_pretrained(
#             model_args.model_name_or_path,
#             cache_dir=training_args.cache_dir,
#             model_max_length=training_args.model_max_length,
#             padding_side="right"
#         )
#     else:
#         tokenizer = transformers.AutoTokenizer.from_pretrained(
#             model_args.model_name_or_path,
#             cache_dir=training_args.cache_dir,
#             model_max_length=training_args.model_max_length,
#             padding_side="right",
#             use_fast=False,
#         )

#     if model_args.version == "v0":
#         if tokenizer.pad_token is None:
#             smart_tokenizer_and_embedding_resize(
#                 special_tokens_dict=dict(pad_token="[PAD]"),
#                 tokenizer=tokenizer,
#                 model=model,
#             )
#     elif model_args.version == "v0.5":
#         tokenizer.pad_token = tokenizer.unk_token
#     else:
#         tokenizer.pad_token = tokenizer.unk_token
#         if model_args.version in conversation_lib.conv_templates:
#             conversation_lib.default_conversation = conversation_lib.conv_templates[
#                 model_args.version]
#         else:
#             conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

#     if model_args.vision_tower is not None:
#         model.get_model().initialize_vision_modules(
#             model_args=model_args,
#             fsdp=training_args.fsdp
#         )

#         vision_tower = model.get_vision_tower()
#         vision_tower.to(
#             dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

#         data_args.image_processor = vision_tower.image_processor
#         data_args.is_multimodal = True

#         model.config.image_aspect_ratio = data_args.image_aspect_ratio
#         model.config.tokenizer_padding_side = tokenizer.padding_side
#         model.config.tokenizer_model_max_length = tokenizer.model_max_length

#         model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
#         if model_args.tune_mm_mlp_adapter:
#             model.requires_grad_(False)
#             for p in model.get_model().mm_projector.parameters():
#                 p.requires_grad = True

#         model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
#         if training_args.freeze_mm_mlp_adapter:
#             for p in model.get_model().mm_projector.parameters():
#                 p.requires_grad = False

#         if training_args.bits in [4, 8]:
#             model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

#         model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
#         model.config.mm_projector_lr = training_args.mm_projector_lr
#         training_args.use_im_start_end = model_args.mm_use_im_start_end
#         model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
#         model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

#     if training_args.bits in [4, 8]:
#         from peft.tuners.lora import LoraLayer
#         for name, module in model.named_modules():
#             if isinstance(module, LoraLayer):
#                 if training_args.bf16:
#                     module = module.to(torch.bfloat16)
#             if 'norm' in name:
#                 module = module.to(torch.float32)
#             if 'lm_head' in name or 'embed_tokens' in name:
#                 if hasattr(module, 'weight'):
#                     if training_args.bf16 and module.weight.dtype == torch.float32:
#                         module = module.to(torch.bfloat16)

#     data_module = make_supervised_data_module(tokenizer=tokenizer,
#                                               data_args=data_args)
#     trainer = LLaVATrainer(model=model,
#                            tokenizer=tokenizer,
#                            args=training_args,
#                            **data_module)

#     if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
#         trainer.train(resume_from_checkpoint=True)
#     else:
#         trainer.train()
#     trainer.save_state()

#     model.config.use_cache = True

#     if training_args.lora_enable:
#         state_dict = get_peft_state_maybe_zero_3(
#             model.named_parameters(), training_args.lora_bias
#         )
#         non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
#             model.named_parameters()
#         )
#         if training_args.local_rank == 0 or training_args.local_rank == -1:
#             model.config.save_pretrained(training_args.output_dir)
#             model.save_pretrained(
#                 training_args.output_dir, state_dict=state_dict)
#             torch.save(non_lora_state_dict, os.path.join(
#                 training_args.output_dir, 'non_lora_trainables.bin'))
#     else:
#         safe_save_model_for_hf_trainer(trainer=trainer,
#                                        output_dir=training_args.output_dir)


# if __name__ == "__main__":
#     train()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
  parser.add_argument("--model-base", type=str, default=None)
  parser.add_argument("--image-file", type=str, required=True)
  parser.add_argument("--device", type=str, default="cuda")
  parser.add_argument("--conv-mode", type=str, default=None)
  parser.add_argument("--temperature", type=float, default=0.2)
  parser.add_argument("--max-new-tokens", type=int, default=512)
  parser.add_argument("--load-8bit", action="store_true")
  parser.add_argument("--load-4bit", action="store_true")
  parser.add_argument("--debug", action="store_true")
  args = parser.parse_args()
  main(args)
