from peft import LoraConfig, AdaLoraConfig, TaskType, IA3Config, AdaptionPromptConfig, VeraConfig, LNTuningConfig, get_peft_model, tuners
from constants_loader import get_constant_list

SENTENCE_TRANSFORMER_TASK = get_constant_list("SENTENCE_TRANSFORMER")
CROSS_ENCODER_TASK = get_constant_list("CROSS_ENCODER")
MULTIMODAL_LANGUAGE_MODEL = get_constant_list("MULTIMODAL_LANGUAGE_MODEL")
CLASSIFICATION_TASK = get_constant_list("CLASSIFICATION")

def get_peft_config(peftconfig_args, model_train_method):
    if model_train_method == 'clm':
        task_type = TaskType.CAUSAL_LM
    elif model_train_method == CLASSIFICATION_TASK:
        task_type = TaskType.SEQ_CLS
    elif model_train_method in CROSS_ENCODER_TASK:
        task_type = TaskType.SEQ_CLS
    elif model_train_method in MULTIMODAL_LANGUAGE_MODEL:
        task_type = TaskType.CAUSAL_LM
    else:
        raise ValueError("[FATAL ERROR] Unexpected Method Selected. Please Check 'Train Method'")

    if peftconfig_args.peft_type == "lora":
        peft_config = LoraConfig(
            r=peftconfig_args.lora_r,
            lora_alpha=peftconfig_args.lora_alpha,
            lora_dropout=peftconfig_args.lora_dropout,
            target_modules=peftconfig_args.lora_target_modules,
            modules_to_save=peftconfig_args.lora_modules_to_save,
            bias="none",
            task_type=task_type,
        )

    elif peftconfig_args.peft_type == "adalora":
        peft_config = AdaLoraConfig(
            init_r=peftconfig_args.adalora_init_r,
            target_r=peftconfig_args.adalora_target_r,
            tinit=peftconfig_args.adalora_tinit,
            tfinal=peftconfig_args.adalora_tfinal,
            deltaT=peftconfig_args.adalora_delta_t,
            lora_alpha=peftconfig_args.lora_alpha,
            lora_dropout=peftconfig_args.lora_dropout,
            target_modules=peftconfig_args.lora_target_modules,
            modules_to_save=peftconfig_args.lora_modules_to_save,
            orth_reg_weight=peftconfig_args.adalora_orth_reg_weight,
            bias="none",
            task_type=task_type,
        )

    elif peftconfig_args.peft_type == "ia3":
        peft_config = IA3Config(
            target_modules=peftconfig_args.ia3_target_modules,
            feedforward_modules=peftconfig_args.feedforward_modules,
            modules_to_save=peftconfig_args.lora_modules_to_save,
            task_type=task_type,
        )
    elif peftconfig_args.peft_type == "llama-adapter":
        peft_config = AdaptionPromptConfig(
            adapter_layers=peftconfig_args.adapter_layers,
            adapter_len=peftconfig_args.adapter_len,
            task_type=task_type,
        )

    elif peftconfig_args.peft_type == "vera":
        peft_config = VeraConfig(
            target_modules=peftconfig_args.vera_target_modules, task_type=task_type, init_weights=False
        )
    elif peftconfig_args.peft_type == "ln_tuning":
        peft_config = LNTuningConfig(
            target_modules=peftconfig_args.ln_target_modules,
            task_type=task_type,
        )

    return peft_config
