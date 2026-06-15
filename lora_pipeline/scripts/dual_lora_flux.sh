ip1=10.201.17.29
ip2=10.201.17.36
ip3=10.201.18.49
ip5=10.201.19.61

output_meta_root=s3://lanjinghong-data/loras_eval_flux_debug_1226
lora_root=s3://collect-data-datasets/202510/civitai_file/'Flux.1 D'
output_root=/mnt/jfs/loras_combine/flux_0321_dual_lora
negative_prompt=""


while true; do
    # output_root=/mnt/jfs/loras_combine/flux_0215/
    # negative_prompt=""
    # prompt_txt=/data/benchmark_metrics/lora_pipeline/meta/prompts/NULL_PROMPT.txt
    # pair_model_id_txt=/data/benchmark_metrics/lora_pipeline/meta/model_ids/flux_content_x_flux0325_style_pairs.txt
    # pair_model_id_txt=/data/benchmark_metrics/logs/triplet_style_firsthit_judge_0325_0.5_2/style_firsthit_non_empty_keys.txt
    # num_prompts=10
    # python /data/benchmark_metrics/lora_pipeline/dual_lora_flux.py \
    #     --lora-root "$lora_root" \
    #     --meta-root "$output_meta_root" \
    #     --output-root "$output_root" \
    #     --pair-model-id-txt "$pair_model_id_txt" \
    #     --base-model flux1-dev.safetensors \
    #     --workflow-json /data/benchmark_metrics/lora_pipeline/meta/workflows/flux_dual_lora.json \
    #     --prompt-txt "" \
    #     --comfy-host http://$ip1,http://$ip2,http://$ip3,http://$ip4,http://$ip5,http://$ip6,http://$ip7,http://$ip8,http://$ip9,http://$ip10,http://$ip11,http://$ip12,http://$ip13,http://$ip14,http://$ip15,http://$ip16,http://$ip17,http://$ip18,http://$ip19,http://$ip20,http://$ip21,http://$ip22,http://$ip23,http://$ip24,http://$ip25,http://$ip26,http://$ip27,http://$ip28 \
    #     --num-workers 8 \
    #     --download-retry-rounds 4 \
    #     --download-retry-wait 3 \
    #     --download-workers 4 \
    #     --num-prompts $num_prompts \
    #     --prefix-phrase "solo" \
    #     --negative-prompt "$negative_prompt" \
    #     --allow-empty-prompt-body
    num_prompts=10
    output_root=/mnt/jfs/loras_combine/flux_0323_dual_lora_diverse_save_prompt
    prompt_txt=/data/benchmark_metrics/lora_pipeline/meta/prompts/OTHER_UNIVERSE_TRIGGER.txt
    pair_model_id_txt=/data/benchmark_metrics/lora_pipeline/meta/model_ids/flux_style_and_content.txt
    pair_model_id_txt=/data/benchmark_metrics/logs/triplet_style_firsthit_judge_0325_0.5_2/style_firsthit_non_empty_keys.txt
    pair_model_id_txt=/data/benchmark_metrics/lora_pipeline/meta/model_ids/flux_content_sample__x__selections_with_origin_style_flux0325_keys.txt 
    pair_model_id_txt=/data/benchmark_metrics/logs/triplet_style_firsthit_judge_0325_0.5_2/true_pair.txt
    pair_model_id_txt=/data/benchmark_metrics/logs/triplet_style_firsthit_judge_0325_0.5_2_2match/style_firsthit_non_empty_keys.txt
    pair_model_id_txt=/data/benchmark_metrics/logs/triplet_style_firsthit_judge_0325_0.5_2_2match/flux_0323_dual_lora_diverse_save_prompt_0328_lt10_unfinished_pair_model_ids.txt
    pair_model_id_txt=/data/benchmark_metrics/logs/triplet_style_firsthit_judge_0325_0.55_2_2match_global_judge/style_firsthit_nonempty_keys_img_lt10_flux_0323_dual_lora_diverse_save_prompt_0328.txt
    output_root=/mnt/jfs/loras_combine/flux_0323_dual_lora_diverse_save_prompt_0328
    python /data/benchmark_metrics/lora_pipeline/dual_lora_flux.py \
        --lora-root "$lora_root" \
        --meta-root "$output_meta_root" \
        --output-root "$output_root" \
        --pair-model-id-txt "$pair_model_id_txt" \
        --base-model flux1-dev.safetensors \
        --workflow-json /data/benchmark_metrics/lora_pipeline/meta/workflows/flux_dual_lora.json \
        --prompt-txt "$prompt_txt" \
        --comfy-host http://$ip1,http://$ip2,http://$ip3,http://$ip4,http://$ip5 \
        --num-workers 8 \
        --download-retry-rounds 4 \
        --download-retry-wait 3 \
        --download-workers 4 \
        --num-prompts $num_prompts \
        --prefix-phrase "solo" \
        --negative-prompt "$negative_prompt"
done
