comfy_hosts=(
    "http://10.201.16.4"
    "http://10.201.16.5"
    "http://10.201.16.11"
    "http://10.201.16.34"
    "http://10.201.16.49"
    "http://10.201.16.50"
    "http://10.201.16.54"
    "http://10.201.16.56"
    "http://10.201.16.63"
    "http://10.201.16.64"
    "http://10.201.17.29"
    "http://10.201.17.33"
    "http://10.201.17.34"
    "http://10.201.17.36"
    "http://10.201.17.43"
    "http://10.201.17.53"
    "http://10.201.17.54"
    "http://10.201.17.58"
    "http://10.201.17.59"
    "http://10.201.17.65"
    "http://10.201.18.6"
    "http://10.201.18.28"
    "http://10.201.18.41"
    "http://10.201.18.49"
    "http://10.201.18.53"
    "http://10.201.19.16"
    "http://10.201.19.23"
    "http://10.201.19.28"
    "http://10.201.19.33"
    "http://10.201.19.39"
    "http://10.201.19.41"
    "http://10.201.19.53"
    "http://10.201.19.61"
)
comfy_host_csv="$(IFS=,; echo "${comfy_hosts[*]}")"
output_meta_root=s3://lanjinghong-data/loras_eval_qwen
lora_root=/mnt/jfs/all_loras/civitai/Qwen
output_root=/mnt/jfs/loras_combine/qwen_0323_dual_lora
prompt_txt=/data/benchmark_metrics/lora_pipeline/meta/prompts/STYLE_UNIVERSE_TRIGGER.txt
while true;do
    python /data/benchmark_metrics/lora_pipeline/dual_lora_qwen.py \
    --lora-root $lora_root \
    --meta-root $output_meta_root \
    --output-root $output_root \
    --pair-model-id-txt /data/benchmark_metrics/lora_pipeline/meta/model_ids/qwen_style_and_content.txt \
    --base-model qwen_image_fp8_e4m3fn.safetensors \
    --prompt-txt $prompt_txt \
    --comfy-host "$comfy_host_csv" \
    --num-workers 4 \
    --num-prompts 100 \
    --download-workers 4 \
    --negative-prompt ""
done
