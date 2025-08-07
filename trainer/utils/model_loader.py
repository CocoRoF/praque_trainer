import os
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoModelForImageTextToText,
)
from sentence_transformers import (
    models,
    SentenceTransformer,
)
from trainer.utils.constants_loader import get_constant_list

TEXT_EMBEDDING_TASK = get_constant_list("TEXT_EMBEDDING")
SENTENCE_TRANSFORMER_TASK = get_constant_list("SENTENCE_TRANSFORMER")
CROSS_ENCODER_TASK = get_constant_list("CROSS_ENCODER")
MULTIMODAL_LANGUAGE_MODEL = get_constant_list("MULTIMODAL_LANGUAGE_MODEL")

def huggingface_model_load(model_path, task, num_labels, st_model_arg, max_seq_length, pooling_mode, dense_feature, token=None, device=None, language_model_class=None, **args):
    print("[INFO] Model Load Started...")
    if language_model_class == "gemma3":
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, num_labels=num_labels, token=token, **args
        )
        print("[INFO] Gemma3 Model Selected. Using 'AutoModelForImageTextToText'.")

    elif language_model_class == "qwen3":
        model = AutoModelForCausalLM.from_pretrained(
            model_path, num_labels=num_labels, token=token, **args
        )
        print("[INFO] Qwen3 Model Selected. Using 'AutoModelForCausalLM'.")

    else:
        print(f"[FATAL ERROR!] `{task}` is not Valid!")
        print(f"SENTENCE_TRANSFORMER =", SENTENCE_TRANSFORMER_TASK)
        print(f"CROSS_ENCODER =", CROSS_ENCODER_TASK)
        print(f"MULTIMODAL_LANGUAGE_MODEL =", MULTIMODAL_LANGUAGE_MODEL)
        raise ValueError("[FATAL ERROR] Method Undefined. Fail to load Model.")

    return model
