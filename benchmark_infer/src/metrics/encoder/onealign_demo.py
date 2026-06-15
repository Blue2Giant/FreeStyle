import torch
from transformers import AutoModelForCausalLM
from transformers.cache_utils import Cache

model = AutoModelForCausalLM.from_pretrained("/mnt/jfs/model_zoo/one-align", trust_remote_code=True, attn_implementation="eager", 
                                             torch_dtype=torch.float16, device_map="auto",
                                             use_cache=False)

from PIL import Image
image = Image.open("/data/LoraPipeline/output/gpt4o_judge/jiegeng.png")
score = model.score([image], task_="aesthetics", input_="image")
print(score)
# task_ : quality | aesthetics; # input_: image | video