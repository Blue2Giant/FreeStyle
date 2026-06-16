ip1=10.201.17.29
ip2=10.201.17.36

meta_root="download from https://huggingface.co/datasets/Blue2Giant/free_style_lora_meta, choose the flux subfolder"
lora_root="/mnt/jfs/flux"  #downlaod frmo civitai, place in a directory, and point the path to it
output_root=""
negative_prompt=""

while true; do
    num_prompts=10
    output_root=""
    prompt_txt=../../meta/prompts/OTHER_UNIVERSE_TRIGGER.txt
    pair_model_id_txt=../meta/model_ids/classified/flux__dual_lora.txt
    python /data/benchmark_metrics/lora_pipeline/dual_lora_flux.py \
        --lora-root "$lora_root" \
        --meta-root "$output_meta_root" \
        --output-root "$output_root" \
        --pair-model-id-txt "$pair_model_id_txt" \
        --base-model flux1-dev.safetensors \
        --workflow-json /data/benchmark_metrics/lora_pipeline/meta/workflows/flux_dual_lora.json \
        --prompt-txt "$prompt_txt" \
        --comfy-host http://$ip1,http://$ip2 \
        --num-workers 8 \
        --download-retry-rounds 4 \
        --download-retry-wait 3 \
        --download-workers 4 \
        --num-prompts $num_prompts \
        --prefix-phrase "solo" \
        --negative-prompt "$negative_prompt"
done
