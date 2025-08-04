def filter_unuse_parms(training_args):
    args_dict = vars(training_args).copy()
    for key in ["distributed_state", "_n_gpu", "__cached__setup_devices", "deepspeed_plugin"]:
        args_dict.pop(key, None)
        
    return args_dict

def set_additional_parms(training_args, base_args, data_args, peftconfig_args, add_trainer_args):
    training_args.ddp_timeout = 3600
    
    if base_args.use_dpotrainer and peftconfig_args.use_peft:
        training_args.model_adapter_name = 'default'
        training_args.ref_adapter_name = 'reference'
    
    if base_args.use_dpotrainer:
        training_args.loss_type = add_trainer_args.dpo_loss_type
        training_args.beta = add_trainer_args.dpo_beta
        training_args.lebel_smoothing = add_trainer_args.dpo_label_smoothing
        training_args.max_length = data_args.tokenizer_max_len
        training_args.max_prompt_length = (data_args.tokenizer_max_len // 2)
        
    try:
        import torch
        import os
        num_gpus = torch.cuda.device_count()
        try:
            proc_num = (os.cpu_count() // num_gpus)
        except:
            proc_num = (os.cpu_count() // 4)
        training_args.dataset_num_proc = proc_num
        
        print(f"[TRAINER] Set dataset_num_proc: {training_args.dataset_num_proc}")
    except:
        ...
    
    return training_args