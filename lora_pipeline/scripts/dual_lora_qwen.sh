comfy_hosts=(
    "http://10.201.16.4"
    "http://10.201.16.5"
)
comfy_host_csv="$(IFS=,; echo "${comfy_hosts[*]}")"
output_meta_root="download from https://huggingface.co/datasets/Blue2Giant/free_style_lora_meta choose the qwen subfolder"
lora_root="/mnt/jfs/Qwen"  #downlaod frmo civitai, place in a directory, and point the path to it
output_root=/mnt/jfs/loras_combine/qwen_0323_dual_lora
prompt_txt=../../meta/prompts/STYLE_UNIVERSE_TRIGGER.txt
while true;do
    python /data/benchmark_metrics/lora_pipeline/dual_lora_qwen.py \
    --lora-root $lora_root \
    --meta-root $output_meta_root \
    --output-root $output_root \
    --pair-model-id-txt ../meta/model_ids/classified/qwen_dual_lora.txt \
    --base-model qwen_image_fp8_e4m3fn.safetensors \
    --prompt-txt $prompt_txt \
    --comfy-host "$comfy_host_csv" \
    --num-workers 4 \
    --num-prompts 100 \
    --download-workers 4 \
    --negative-prompt ""
done
