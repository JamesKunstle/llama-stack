curl -H 'Content-Type: application/json' \
	-X POST \
	-d '{"job_uuid": "1234", "model": "granite", "training_config": {"n_epochs": 1, "max_steps_per_epoch": 1, "gradient_accumulation_steps": 1, "max_validation_steps": 1, "data_config": {"dataset_id": "None", "batch_size": 1, "shuffle": false, "data_format": "instruct"}, "optimizer_config": {"optimizer_type": "adamw", "lr": 0.0005, "weight_decay": 0, "num_warmup_steps": 1}}, "hyperparam_search_config": {}, "logger_config": {}, "algorithm_config": {"type": "LoRA", "lora_attn_modules": [], "apply_lora_to_mlp": false, "apply_lora_to_output": false, "rank": 0, "alpha": 0}, "checkpoint_dir": "/dev/null"}' \
	http://localhost:8321/v1/post-training/supervised-fine-tune | jq
