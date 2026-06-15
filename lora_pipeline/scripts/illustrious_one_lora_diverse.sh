#l40s
ip1=10.201.16.63
ip2=10.201.16.50
ip3=10.201.17.59
ip4=10.201.16.54
ip5=10.201.18.49
ip6=10.201.17.65
ip7=10.201.17.54
ip8=10.201.17.34
ip9=10.201.17.58
ip10=10.201.18.28
ip11=10.201.19.23
ip12=10.201.19.53
ip13=10.201.19.41
ip14=10.201.19.33
ip15=10.201.18.8
ip16=10.201.16.34
ip17=10.201.19.28
ip18=10.201.16.61
ip19=10.201.18.41
ip20=10.201.17.43
ip21=10.201.17.66
ip22=10.201.16.49
ip23=10.201.17.33
ip24=10.201.19.16
ip25=10.201.16.5
ip26=10.201.16.11
ip27=10.201.17.36

# civitai_illustrious_loras="s3://collect-data-datasets/202510/civitai_file/illustrious_1223/"
output_civitai_illustrious=s3://lanjinghong-data/loras_eval_illustrious_one_img_magic
civitai_illustrious_loras=/mnt/jfs/all_loras/civitai/Illustrious/
output_root=/mnt/jfs/loras_combine/illustrious_0321_two_lora
character_prompt_txt=/data/LoraPipeline/scripts/CHARACTER_UNIVERSE_TRIGGER.txt
other_prompt_txt=/data/LoraPipeline/scripts/OTHER_UNIVERSE_TRIGGER.txt
style_prompt_txt=/data/LoraPipeline/scripts/STYLE_UNIVERSE_TRIGGER.txt
num_prompts=20
negative_prompt="lowres, normal quality, worst quality, low quality, jpeg artifacts, compression artifacts, pixelated, blurry, out of focus, soft focus, bad contrast, color banding, posterization, chromatic aberration, aliasing, moire, overexposed, underexposed, blown highlights, crushed shadows, noise, watermark, logo, text, caption, signature, username, copyright, bad anatomy, malformed, disfigured, deformed, bad proportions, extra limbs, missing limbs, duplicate body parts, extra digits, missing fingers, fused fingers, webbed fingers, bad hands, bad feet, distorted face, asymmetrical eyes, cross-eye, extra face, cloned person, body cut off, cropped, floating objects, disconnected limbs, perspective errors, depth errors, incorrect shadows, inconsistent lighting, repeated patterns, mirror artifacts"
negative_prompt=""
while true;do
    python /data/benchmark_metrics/lora_pipeline/illustrious_one_lora_diverse.py \
        --lora-root "$civitai_illustrious_loras" \
        --meta-root "$output_civitai_illustrious" \
        --output-root "$output_root" \
        --base-model Illustrious-XL-v1.0.safetensors \
        --filter-model-id /data/LoraPipeline/scripts/character_illustrious.txt \
        --comfy-host http://$ip1,http://$ip2,http://$ip3,http://$ip4,http://$ip5,http://$ip6,http://$ip7,http://$ip8,http://$ip9,http://$ip10,http://$ip11,http://$ip12,http://$ip13,http://$ip14,http://$ip15,http://$ip16,http://$ip17,http://$ip18,http://$ip19,http://$ip20,http://$ip21,http://$ip22,http://$ip23,http://$ip24,http://$ip25,http://$ip26,http://$ip27 \
        --workflow-json /data/LoraPipeline/assets/illustrious_one_lora.json \
        --prompt-txt "$character_prompt_txt" \
        --num-workers 8 \
        --download-retry-rounds 4 \
        --download-retry-wait 3 \
        --num-prompts $num_prompts \
        --prefix-phrase "solo" \
        --negative-prompt "$negative_prompt" \
        --negative-node-id 12

    python /data/benchmark_metrics/lora_pipeline/illustrious_one_lora_diverse.py \
        --lora-root "$civitai_illustrious_loras" \
        --meta-root "$output_civitai_illustrious" \
        --output-root "$output_root" \
        --base-model Illustrious-XL-v1.0.safetensors \
        --filter-model-id /data/LoraPipeline/scripts/others_illustrious.txt\
        --comfy-host http://$ip1,http://$ip2,http://$ip3,http://$ip4,http://$ip5,http://$ip6,http://$ip7,http://$ip8,http://$ip9,http://$ip10,http://$ip11,http://$ip12,http://$ip13,http://$ip14,http://$ip15,http://$ip16,http://$ip17,http://$ip18,http://$ip19,http://$ip20,http://$ip21,http://$ip22,http://$ip23,http://$ip24,http://$ip25,http://$ip26,http://$ip27 \
        --workflow-json /data/LoraPipeline/assets/illustrious_one_lora.json  \
        --prompt-txt "$other_prompt_txt" \
        --num-workers 8 \
        --download-retry-rounds 4 \
        --download-retry-wait 3 \
        --num-prompts $num_prompts \
        --prefix-phrase "" \
        --negative-prompt "$negative_prompt" \
        --negative-node-id 12
    python /data/benchmark_metrics/lora_pipeline/illustrious_one_lora_diverse.py \
        --lora-root "$civitai_illustrious_loras" \
        --meta-root "$output_civitai_illustrious" \
        --output-root "$output_root" \
        --base-model Illustrious-XL-v1.0.safetensors \
        --filter-model-id /data/LoraPipeline/assets/illustrious_style_ids.txt \
        --workflow-json /data/LoraPipeline/assets/illustrious_one_lora.json \
        --comfy-host http://$ip1,http://$ip2,http://$ip3,http://$ip4,http://$ip5,http://$ip6,http://$ip7,http://$ip8,http://$ip9,http://$ip10,http://$ip11,http://$ip12,http://$ip13,http://$ip14,http://$ip15,http://$ip16,http://$ip17,http://$ip18,http://$ip19,http://$ip20,http://$ip21,http://$ip22,http://$ip23,http://$ip24,http://$ip25,http://$ip26,http://$ip27 \
        --prompt-txt "$style_prompt_txt" \
        --num-workers 8 \
        --download-retry-rounds 4 \
        --download-retry-wait 3 \
        --num-prompts $num_prompts \
        --prefix-phrase "" \
        --negative-prompt "$negative_prompt" \
        --negative-node-id 12 \
        --overwrite

done
#        --filter-model-id /data/LoraPipeline/assets/illutrious_for_multi_persion.txt \
