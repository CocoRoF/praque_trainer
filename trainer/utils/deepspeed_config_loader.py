import json

from constants_loader import get_constant_list

SENTENCE_TRANSFORMER_TASK = get_constant_list("SENTENCE_TRANSFORMER")
CROSS_ENCODER_TASK = get_constant_list("CROSS_ENCODER")
MULTIMODAL_LANGUAGE_MODEL = get_constant_list("MULTIMODAL_LANGUAGE_MODEL")

def get_deepspeed_config(training_args, deepspeed_args):
    if (deepspeed_args.ds_jsonpath is not None) and (len(deepspeed_args.ds_jsonpath) > 0):
        with open(deepspeed_args.ds_jsonpath, "r") as f:
            deepspeed_config = json.load(f)

    elif deepspeed_args.ds_preset == 'zero-1':
        print("[INFO] USE DeepSpeed: Zero Optimization Stage 1")
        deepspeed_config = {
            "train_batch_size": "auto",
            "gradient_accumulation_steps": "auto",
            "zero_optimization": {
                "stage": 1,
            },
            "fp16": {
                "enabled": training_args.fp16
            },
            "bf16": {
                "enabled": training_args.bf16
            },
            "gradient_clipping": training_args.max_grad_norm
        }

    elif deepspeed_args.ds_preset == 'zero-2':
        print("[INFO] USE DeepSpeed: Zero Optimization Stage 2")

        if training_args.gradient_checkpointing:
            print("[INFO] Gradient Checkpointing Enabled — Disabling Offload Options for Compatibility")
            offload_optimizer = {
                "device": "none",
                "pin_memory": False
            }
            offload_param = {
                "device": "none",
                "pin_memory": False
            }
        else:
            offload_optimizer = {
                "device": "cpu",
                "pin_memory": True
            }
            offload_param = {
                "device": "cpu",
                "pin_memory": True
            }
        deepspeed_config = {
            "train_batch_size": "auto",
            "gradient_accumulation_steps": "auto",
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": offload_optimizer,
                "offload_param": offload_param,
                "allgather_partitions": True,
                "allgather_bucket_size": deepspeed_args.ds_stage2_bucket_size,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": deepspeed_args.ds_stage2_bucket_size,
                "contiguous_gradients": True,
                "round_robin_gradients": True,
            },
            "fp16": {
                "enabled": training_args.fp16
            },
            "bf16": {
                "enabled": training_args.bf16
            },
            "gradient_clipping": training_args.max_grad_norm
        }

    elif deepspeed_args.ds_preset == 'zero-2-non-offload':
        print("[INFO] USE DeepSpeed: Zero Optimization Stage 2 (Non-Offload)")
        deepspeed_config = {
            "train_batch_size": "auto",
            "gradient_accumulation_steps": "auto",
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": {
                    "device": "none",
                    "pin_memory": False
                },
                "offload_param": {
                    "device": "none",
                    "pin_memory": False
                },
                "allgather_partitions": True,
                "allgather_bucket_size": deepspeed_args.ds_stage2_bucket_size,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": deepspeed_args.ds_stage2_bucket_size,
                "contiguous_gradients": True,
                "round_robin_gradients": True,
            },
            "fp16": {
                "enabled": training_args.fp16
            },
            "bf16": {
                "enabled": training_args.bf16
            },
            "gradient_clipping": training_args.max_grad_norm
        }

    elif deepspeed_args.ds_preset == 'zero-3':
        print("[INFO] USE DeepSpeed: Zero Optimization Stage 3")

        if training_args.gradient_checkpointing:
            print("[INFO] Gradient Checkpointing Enabled — Disabling Offload Options for Compatibility (ZeRO-3)")
            offload_optimizer = {
                "device": "cpu",
                "pin_memory": True
            }
            offload_param = {
                "device": "cpu",
                "pin_memory": True
            }
        else:
            offload_optimizer = {
                "device": "cpu",
                "pin_memory": True
            }
            offload_param = {
                "device": "cpu",
                "pin_memory": True
            }
        deepspeed_config = {
            "train_batch_size": "auto",
            "gradient_accumulation_steps": "auto",
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": offload_optimizer,
                "offload_param": offload_param,
                "overlap_comm": True,
                "contiguous_gradients": True,
                "sub_group_size": deepspeed_args.ds_stage3_sub_group_size,
                "reduce_bucket_size": "auto",
                "stage3_prefetch_bucket_size": "auto",
                "stage3_param_persistence_threshold": "auto",
                "stage3_max_live_parameters": deepspeed_args.ds_stage3_max_live_parameters,
                "stage3_max_reuse_distance": deepspeed_args.ds_stage3_max_reuse_distance,
                "stage3_gather_16bit_weights_on_model_save": True,
            },
            "fp16": {
                "enabled": training_args.fp16
            },
            "bf16": {
                "enabled": training_args.bf16
            },
            "gradient_clipping": training_args.max_grad_norm
        }
    else:
        print("[FATAL ERROR] DeepSpeed Config is Missing!")
        print("[FATAL ERROR] Can't Find 'deepspeed_config.json' File")
        print("[FATAL ERROR] 'zero-1', 'zero-2', 'zero-3' are only available for ds_preset")
        raise RuntimeError

    return deepspeed_config
