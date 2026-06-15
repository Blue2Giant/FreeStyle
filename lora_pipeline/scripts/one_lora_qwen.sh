# qwen one lora
ip2=10.201.17.60
ip1=10.191.13.9

output_meta_root=s3://lanjinghong-data/loras_eval_qwen
lora_root=/mnt/jfs/all_loras/civitai/Qwen
output_root=/mnt/jfs/loras_combine/qwen_0323_one_lora
character_prompt_txt=/data/benchmark_metrics/lora_pipeline/meta/prompts/CHARACTER_UNIVERSE_TRIGGER.txt
other_prompt_txt=/data/benchmark_metrics/lora_pipeline/meta/prompts/OTHER_UNIVERSE_TRIGGER.txt
style_prompt_txt=/data/benchmark_metrics/lora_pipeline/meta/prompts/STYLE_UNIVERSE_TRIGGER.txt
num_prompts=20
negative_prompt=""

while true; do
    python /data/benchmark_metrics/lora_pipeline/one_lora_qwen.py \
        --lora-root "$lora_root" \
        --meta-root "$output_meta_root" \
        --output-root "$output_root" \
        --base-model qwen_image_fp8_e4m3fn.safetensors \
        --workflow-json /data/benchmark_metrics/lora_pipeline/meta/workflows/qwen_one_lora0320.json \
        --prompt-txt "$style_prompt_txt" \
        --filter-model-id /data/benchmark_metrics/lora_pipeline/meta/model_ids/qwen_style.txt \
        --comfy-host http://$ip1,http://$ip2 \
        --num-workers 4 \
        --download-retry-rounds 4 \
        --download-retry-wait 3 \
        --download-workers 4 \
        --num-prompts $num_prompts \
        --prefix-phrase "" \
        --negative-prompt "$negative_prompt"
    python /data/benchmark_metrics/lora_pipeline/one_lora_qwen.py \
        --lora-root "$lora_root" \
        --meta-root "$output_meta_root" \
        --output-root "$output_root" \
        --base-model qwen_image_fp8_e4m3fn.safetensors \
        --workflow-json /data/benchmark_metrics/lora_pipeline/meta/workflows/qwen_one_lora0320.json \
        --prompt-txt "$character_prompt_txt" \
        --filter-model-id /data/benchmark_metrics/lora_pipeline/meta/model_ids/character_qwen.txt \
        --comfy-host http://$ip1,http://$ip2 \
        --num-workers 4 \
        --download-retry-rounds 4 \
        --download-retry-wait 3 \
        --download-workers 4 \
        --num-prompts $num_prompts \
        --prefix-phrase "solo" \
        --negative-prompt "$negative_prompt"
    python /data/benchmark_metrics/lora_pipeline/one_lora_qwen.py \
        --lora-root "$lora_root" \
        --meta-root "$output_meta_root" \
        --output-root "$output_root" \
        --base-model qwen_image_fp8_e4m3fn.safetensors \
        --workflow-json /data/benchmark_metrics/lora_pipeline/meta/workflows/qwen_one_lora0320.json \
        --prompt-txt "$other_prompt_txt" \
        --filter-model-id /data/benchmark_metrics/lora_pipeline/meta/model_ids/other_qwen.txt \
        --comfy-host http://$ip1,http://$ip2 \
        --num-workers 4 \
        --download-retry-rounds 4 \
        --download-retry-wait 3 \
        --download-workers 4 \
        --num-prompts $num_prompts \
        --prefix-phrase "" \
        --negative-prompt "$negative_prompt"
    
done
