#l40s
ip1=10.201.16.63
ip2=10.201.16.50

output_civitai_illustrious=""
civitai_illustrious_loras=/mnt/jfs/Illustrious  #downlaod from civitai, place in a directory, and point the path to it

output_root=/mnt/jfs/loras_combine/illustrious_0321_two_lora
character_prompt_txt=../../meta/prompts/CHARACTER_UNIVERSE_TRIGGER.txt
other_prompt_txt=../../meta/prompts/OTHER_UNIVERSE_TRIGGER.txt
style_prompt_txt=../../meta/prompts/STYLE_UNIVERSE_TRIGGER.txt
num_prompts=20
negative_prompt="lowres, normal quality, worst quality, low quality, jpeg artifacts, compression artifacts, pixelated, blurry, out of focus, soft focus, bad contrast, color banding, posterization, chromatic aberration, aliasing, moire, overexposed, underexposed, blown highlights, crushed shadows, noise, watermark, logo, text, caption, signature, username, copyright, bad anatomy, malformed, disfigured, deformed, bad proportions, extra limbs, missing limbs, duplicate body parts, extra digits, missing fingers, fused fingers, webbed fingers, bad hands, bad feet, distorted face, asymmetrical eyes, cross-eye, extra face, cloned person, body cut off, cropped, floating objects, disconnected limbs, perspective errors, depth errors, incorrect shadows, inconsistent lighting, repeated patterns, mirror artifacts"
negative_prompt=""
while true;do
    python /data/benchmark_metrics/lora_pipeline/illustrious_one_lora_diverse.py \
        --lora-root "$civitai_illustrious_loras" \
        --meta-root "$output_civitai_illustrious" \
        --output-root "$output_root" \
        --base-model Illustrious-XL-v1.0.safetensors \
        --filter-model-id ../../meta/model_ids/classified/character_illustrious.txt \
        --comfy-host http://$ip1,http://$ip2 \
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
        --filter-model-id ../../meta/model_ids/classified/others_illustrious.txt\
        --comfy-host http://$ip1,http://$ip2 \
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
        --filter-model-id ../../meta/model_ids/classified/illustrious_style_ids.txt \
        --workflow-json /data/LoraPipeline/assets/illustrious_one_lora.json \
        --comfy-host http://$ip1,http://$ip2 \
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
