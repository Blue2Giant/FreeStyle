"""
python3 /data/benchmark_metrics/benchmark_metrics/clip_t_demo.py \
  --image /data/benchmark_metrics/assets/content.webp \
  --text "an elf standing" \
  --clip-model /mnt/jfs/model_zoo/openai/clip-vit-base-patch32
"""
"""Calculates the CLIP similarity between one text and one image."""
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--clip-model',
                    type=str,
                    default='openai/clip-vit-base-patch32',
                    help='CLIP model to use')
parser.add_argument('--device',
                    type=str,
                    default=None,
                    help='Device to use. Like cuda, cuda:0 or cpu')
parser.add_argument('--image', type=str, required=True, help='Image path')
parser.add_argument('--text', type=str, required=True, help='Input text')


@torch.no_grad()
def compute_similarity(model, processor, tokenizer, image_path, text, device):
    image = Image.open(image_path).convert('RGB')
    image_inputs = processor(images=image, return_tensors='pt')
    text_inputs = tokenizer(text,
                            return_tensors='pt',
                            padding=True,
                            truncation=True)
    for key in image_inputs:
        image_inputs[key] = image_inputs[key].to(device)
    for key in text_inputs:
        text_inputs[key] = text_inputs[key].to(device)

    image_features = model.get_image_features(**image_inputs)
    text_features = model.get_text_features(**text_inputs)

    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)
    similarity = (image_features * text_features).sum(dim=1)
    return similarity.item()


def main():
    args = parser.parse_args()

    if args.device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(args.device)

    model = AutoModel.from_pretrained(args.clip_model).to(device)
    processor = AutoProcessor.from_pretrained(args.clip_model)
    tokenizer = AutoTokenizer.from_pretrained(args.clip_model)

    score = compute_similarity(model, processor, tokenizer, args.image,
                               args.text, device)
    print('CLIP Similarity: ', score)


if __name__ == '__main__':
    main()
