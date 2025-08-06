import os
from transformers import AutoTokenizer, AutoProcessor

def huggingface_tokenizer_load(model_path, tokenizer_name, max_seq_length, model_subfolder=None, language_model_class=None, **args):
    if language_model_class == "gemma3":
        if (tokenizer_name is not None) and (len(tokenizer_name) > 0):
            try:
                tokenizer = AutoProcessor.from_pretrained(tokenizer_name, use_fast=True)
            except:
                tokenizer = AutoProcessor.from_pretrained(model_path, use_fast=True)
        else:
            tokenizer = AutoProcessor.from_pretrained(model_path, subfolder=model_subfolder, use_fast=True)

        print("[INFO] Gemma3 Model Selected. Using 'AutoProcessor'.")
        return tokenizer

    elif language_model_class == "qwen3":
        if (tokenizer_name is not None) and (len(tokenizer_name) > 0):
            try:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            except:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder=model_subfolder)

        print("[INFO] Qwen3 Model Selected. Using 'AutoTokenizer'.")
        return tokenizer

    else:
        if (tokenizer_name is not None) and (len(tokenizer_name) > 0):
            try:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            except:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder=model_subfolder)

        print("Language model class not specified or not recognized. Using default AutoTokenizer.")
        return tokenizer
