from trainer.utils.optimizer import StableAdamW

def get_optimizer_cls_and_kwargs(model, base_args, training_args):
    if base_args.use_stableadamw:
        optimizer_cls_and_kwargs = (
            StableAdamW,
            {
                "params": model.parameters(),
                "lr": training_args.learning_rate,
                "betas": (training_args.adam_beta1, training_args.adam_beta2),
                "eps": training_args.adam_epsilon,
                "weight_decay": training_args.weight_decay
            }
        )
        
    else:
        ...
        
    return optimizer_cls_and_kwargs

def get_optimizer(model, base_args, training_args):
    if base_args.use_stableadamw:
        optimizer = StableAdamW(
            params=model.parameters(),
            lr=training_args.learning_rate, 
            betas=(training_args.adam_beta1,training_args.adam_beta2), 
            eps=training_args.adam_epsilon, 
            weight_decay=training_args.weight_decay)
        
    else:
        ...
        
    return optimizer