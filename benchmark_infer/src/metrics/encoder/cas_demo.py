"""
python3 /data/benchmark_metrics/benchmark_metrics/calculate_cas.py \
    /data/benchmark_metrics/sample_1500_bench_sref/cref/000000.png \
    /data/benchmark_metrics/sample_1500_bench_sref/sref/000000.png
"""
import argparse
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoConfig
import os

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    # assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1)
    return feat_mean, feat_std

def load_model(device):
    # Try to load from local path first, otherwise use Hugging Face Hub
    local_path = '/mnt/jfs/model_zoo/dinov2-base/'
    model_name = "facebook/dinov2-base"
    
    if os.path.exists(local_path):
        print(f"Loading model from local path: {local_path}")
        model_name_or_path = local_path
    else:
        print(f"Local model path not found, using model name: {model_name}")
        model_name_or_path = model_name

    try:
        config = AutoConfig.from_pretrained(model_name_or_path)
        config.output_hidden_states = True
        processor = AutoImageProcessor.from_pretrained(model_name_or_path)
        model = AutoModel.from_pretrained(model_name_or_path, config=config).to(device)
        model.eval()
        return processor, model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def compute_cas(content_image, style_image, processor, model, device):
    # Ensure images are resized to 512x512 as in the original script
    if content_image.size != (512, 512):
        content_image = content_image.resize((512, 512))
    if style_image.size != (512, 512):
        style_image = style_image.resize((512, 512))

    with torch.no_grad():
        # Process content image
        inputs1 = processor(images=content_image, return_tensors="pt").to(device)
        outputs1 = model(**inputs1)
        image_features1 = outputs1.last_hidden_state
        mean1, std1 = calc_mean_std(image_features1.transpose(-1, -2))
        size1 = image_features1.transpose(-1, -2).size()
        normalized_feat1 = (image_features1.transpose(-1, -2) - mean1.expand(size1)) / std1.expand(size1)

        # Process style image
        inputs2 = processor(images=style_image, return_tensors="pt").to(device)
        outputs2 = model(**inputs2)
        image_features2 = outputs2.last_hidden_state
        mean2, std2 = calc_mean_std(image_features2.transpose(-1, -2))
        size2 = image_features2.transpose(-1, -2).size()
        normalized_feat2 = (image_features2.transpose(-1, -2) - mean2.expand(size2)) / std2.expand(size2)

        # Calculate CAS (MSE between normalized features)
        # Note: In the original script, it was taking mean over dim=(0, 1, 2). 
        # Here we also take mean over all dimensions to get a single scalar score.
        cas = torch.mean((normalized_feat2 - normalized_feat1) ** 2).item()
        return cas

def main():
    parser = argparse.ArgumentParser(description="Calculate CAS similarity between two images.")
    parser.add_argument("image1", type=str, help="Path to the first image (content)")
    parser.add_argument("image2", type=str, help="Path to the second image (style)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model on (cuda or cpu)")
    
    args = parser.parse_args()

    if not os.path.exists(args.image1):
        print(f"Error: Image 1 not found at {args.image1}")
        return
    if not os.path.exists(args.image2):
        print(f"Error: Image 2 not found at {args.image2}")
        return

    device = torch.device(args.device)
    print(f"Using device: {device}")

    processor, model = load_model(device)
    if processor is None or model is None:
        print("Failed to load model.")
        return

    try:
        image1 = Image.open(args.image1).convert("RGB")
        image2 = Image.open(args.image2).convert("RGB")
        
        score = compute_cas(image1, image2, processor, model, device)
        print(f"CAS Similarity Score: {score}")
        
    except Exception as e:
        print(f"Error processing images: {e}")

if __name__ == "__main__":
    main()
