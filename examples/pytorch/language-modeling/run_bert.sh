python3  run_mlm.py \
     --model_name_or_path bert-large-uncased \
     --dataset_name wikitext \
     --dataset_config_name wikitext-2-raw-v1 \
     --do_train \
     --max_steps 16000 \
     --logging_steps 20 \
     --output_dir /tmp/test-mlm-bbu \
     --overwrite_output_dir \
     --per_device_train_batch_size 8 \
     --fp16 \
     --skip_memory_metrics=True \
     --half_precision_backend apex \
     --fp16_backend apex \
     --save_total_limit 5 \
     --load_best_model_at_end=True \
     --evaluation_strategy epoch \
     --save_strategy epoch \
     "$@" \
    2>&1 | tee log.txt
