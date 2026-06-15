python /data/LoraPipeline/scripts/generate_prompts_from_csv.py \
  --input_csv /data/benchmark_metrics/lora_pipeline/meta/prompts/CHARACTER_UNIVERSE_TRIGGER.csv \
  --output_txt /data/benchmark_metrics/lora_pipeline/meta/prompts/CHARACTER_UNIVERSE_TRIGGER.txt \
  --num_prompts 200000 \
  --min_columns 2 \
  --max_columns 5 \
  --min_terms_per_column 0 \
  --max_terms_per_column 2 \
  --has_header \
  --seed 42 \
  --replace-space-with-underscore

python /data/LoraPipeline/scripts/generate_prompts_from_csv.py \
  --input_csv /data/benchmark_metrics/lora_pipeline/meta/prompts/OTHER_UNIVERSE_TRIGGER.csv \
  --output_txt /data/benchmark_metrics/lora_pipeline/meta/prompts/OTHER_UNIVERSE_TRIGGER_UNIQUE_4w.txt \
  --num_prompts 400000 \
  --min_columns 2 \
  --max_columns 5 \
  --min_terms_per_column 0 \
  --max_terms_per_column 2 \
  --has_header \
  --seed 42 \
  --replace-space-with-underscore

python /data/LoraPipeline/scripts/generate_prompts_from_csv.py \
  --input_csv /data/benchmark_metrics/lora_pipeline/meta/prompts/STYLE_UNIVERSE_TRIGGER.csv \
  --output_txt /data/benchmark_metrics/lora_pipeline/meta/prompts/STYLE_UNIVERSE_TRIGGER.txt \
  --num_prompts 200000 \
  --min_columns 2 \
  --max_columns 5 \
  --min_terms_per_column 0 \
  --max_terms_per_column 2 \
  --has_header \
  --seed 42 \
  --replace-space-with-underscore

python /data/LoraPipeline/scripts/generate_prompts_from_csv.py \
  --input_csv /data/benchmark_metrics/lora_pipeline/meta/prompts/STYLE_UNIVERSE_TRIGGER.csv \
  --output_txt /data/benchmark_metrics/lora_pipeline/meta/prompts/TRIPLET_UNIVERSE_TRIGGER.txt \
  --num_prompts 400000 \
  --min_columns 2 \
  --max_columns 5 \
  --min_terms_per_column 0 \
  --max_terms_per_column 2 \
  --has_header \
  --seed 42 \
  --replace-space-with-underscore

python /data/LoraPipeline/scripts/generate_prompts_from_csv.py \
  --input_csv /data/benchmark_metrics/lora_pipeline/meta/prompts/STYLE_UNIVERSE_TRIGGER.csv \
  --output_txt /data/benchmark_metrics/lora_pipeline/meta/prompts/TRIPLET_UNIVERSE_TRIGGER_NO_UNDERLINE.txt \
  --num_prompts 400000 \
  --min_columns 2 \
  --max_columns 5 \
  --min_terms_per_column 0 \
  --max_terms_per_column 2 \
  --has_header \
  --seed 42