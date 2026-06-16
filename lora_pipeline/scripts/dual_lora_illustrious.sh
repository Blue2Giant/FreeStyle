#l40s
ip1=10.201.19.28
ip2=10.201.16.63

output_meta_root=s3://lanjinghong-data/loras_eval_illustrious_one_img_magic
lora_root="/mnt/jfs/Illutrious"  #downlaod frmo civitai, place in a directory, and point the path to it
output_root=""
pair_model_id_txt=../meta/model_ids/classified/illustrious_dual_lora.txt
triplet_prompt_txt=../../meta/prompts/TRIPLET_UNIVERSE_TRIGGER.txt
num_prompts=10
negative_prompt="lowres, normal quality, worst quality, low quality, jpeg artifacts, compression artifacts, pixelated, blurry, out of focus, soft focus, bad contrast, color banding, posterization, chromatic aberration, aliasing, moire, overexposed, underexposed, blown highlights, crushed shadows, noise, watermark, logo, text, caption, signature, username, copyright, bad anatomy, malformed, disfigured, deformed, bad proportions, extra limbs, missing limbs, duplicate body parts, extra digits, missing fingers, fused fingers, webbed fingers, bad hands, bad feet, distorted face, asymmetrical eyes, cross-eye, extra face, cloned person, body cut off, cropped, floating objects, disconnected limbs, perspective errors, depth errors, incorrect shadows, inconsistent lighting, repeated patterns, mirror artifacts"
negative_prompt=""
while true; do
    python /data/benchmark_metrics/lora_pipeline/dual_lora_illustrious.py \
        --lora-root "$lora_root" \
        --meta-root "$output_meta_root" \
        --output-root "$output_root" \
        --pair-model-id-txt "$pair_model_id_txt" \
        --base-model Illustrious-XL-v1.0.safetensors \
        --workflow-json /data/benchmark_metrics/lora_pipeline/meta/workflows/sdxl_dual_lora_ljh.json \
        --prompt-txt "$triplet_prompt_txt" \
        --comfy-host http://$ip1,http://$ip2 \
        --num-workers 8 \
        --download-retry-rounds 4 \
        --download-retry-wait 3 \
        --download-workers 4 \
        --num-prompts $num_prompts \
        --prefix-phrase "solo" \
        --negative-prompt "$negative_prompt"
done
#        --comfy-host http://$ip1,http://$ip2,http://$ip3,http://$ip4,http://$ip5,http://$ip6,http://$ip7,http://$ip8,http://$ip9,http://$ip10,http://$ip11,http://$ip12,http://$ip13,http://$ip14,http://$ip15,http://$ip16,http://$ip17,http://$ip18,http://$ip19,http://$ip20,http://$ip21,http://$ip22,http://$ip23,http://$ip24,http://$ip25,http://$ip26,http://$ip27,http://$ip28,http://$ip29 \
