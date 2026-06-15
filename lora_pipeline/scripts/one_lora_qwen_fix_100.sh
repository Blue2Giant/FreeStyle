# qwen one lora
ip2=10.191.9.11
ip1=10.191.13.9


output_meta_root=s3://lanjinghong-data/loras_eval_qwen
lora_root=s3://collect-data-datasets/202510/civitai_file/Qwen
output_root=/mnt/jfs/loras_combine/qwen_0322_one_lora
prompt_txt=/data/benchmark_metrics/lora_pipeline/meta/prompts/diverse_prompts_100.txt
filter_model_id=/data/benchmark_metrics/lora_pipeline/meta/model_ids/qwen_ids.txt
num_prompts=20
negative_prompt=""

while true; do
    python /data/benchmark_metrics/lora_pipeline/one_lora_qwen.py \
        --lora-root "$lora_root" \
        --meta-root "$output_meta_root" \
        --output-root "$output_root" \
        --base-model qwen_image_fp8_e4m3fn.safetensors \
        --workflow-json /data/benchmark_metrics/lora_pipeline/meta/workflows/qwen_one_lora0320.json \
        --prompt-txt "$prompt_txt" \
        --filter-model-id "$filter_model_id" \
        --comfy-host http://$ip1,http://$ip2 \
        --num-workers 4 \
        --download-retry-rounds 4 \
        --download-retry-wait 3 \
        --download-workers 4 \
        --num-prompts $num_prompts \
        --prefix-phrase "" \
        --negative-prompt "$negative_prompt"
done
