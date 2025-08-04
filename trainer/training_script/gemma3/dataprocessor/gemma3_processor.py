from trainer.utils.dataloader_toolkit import rename_dataset_columns
from trainer.utils.dataloader_toolkit import messages_form_dataset_filter

def gemma_formatter(examples, tokenizer):
    system = examples['system']
    question = examples['question']
    answer = examples['answer']
    if system is None or len(system) < 1:
        system = "You are a helpful assistant."

    if question is None or answer is None:
        raise ValueError("Value Error")

    messages = [
        {"role": "system", "content": [{"type": "text", "text": f"{system}"}]},
        {"role": "user", "content": [{"type": "text", "text": f"{question}"}]},
        {"role": "assistant", "content": [{"type": "text", "text": f"{answer}"}]}
    ]

    text = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False, return_dict=False).split("<bos>")[1]
    return {"text": text}

def gemma3_messages_to_clm_formatter(examples, tokenizer):
    system = examples['messages'][0]['content']
    question = examples['messages'][1]['content']
    answer = examples['messages'][2]['content']

    if system is None or len(system) < 1:
        system = "You are a helpful assistant."

    if question is None or answer is None:
        raise ValueError("Value Error")

    messages = [
        {"role": "system", "content": [{"type": "text", "text": f"{system}"}]},
        {"role": "user", "content": [{"type": "text", "text": f"{question}"}]},
        {"role": "assistant", "content": [{"type": "text", "text": f"{answer}"}]}
    ]

    text = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False, return_dict=False).split("<bos>")[1]
    result = {"text": text}
    return result

def gemma_sft_formatter(examples):
    system = examples['system']
    question = examples['question']
    answer = examples['answer']
    if system is None or len(system) < 1:
        system = "You are a helpful assistant."

    if question is None or answer is None:
        raise ValueError("Value Error")

    messages = [
        {"role": "system", "content": f"{system}"},
        {"role": "user", "content": f"{question}"},
        {"role": "assistant", "content":  f"{answer}"}
    ]

    return {"messages": messages}

def encode(examples, tokenizer, col_name, tokenizer_max_len):
    return tokenizer.tokenizer(examples[col_name], truncation=True, max_length=tokenizer_max_len)

def processor(
    train_dataset,
    test_dataset,
    tokenizer,
    tokenizer_max_len,
    num_proc,
    selected_trainer_name
):
    """
    Process the dataset based on the task type.

    Args:
        train_dataset: The training dataset.
        test_dataset: The testing dataset.
        tokenizer: The tokenizer to use for processing.
        tokenizer_max_len: The maximum length for tokenization.
        num_proc: The number of processes to use for tokenization.

    Returns:
        Processed training and testing datasets.
    """

    if selected_trainer_name == 'none':
        print("[INFO] Default Trainer Selected. Return Dataset with Default Processing.")
        try:
            if train_dataset.column_names == ['messages']:
                train_dataset = messages_form_dataset_filter(train_dataset)
                new_train_dataset = train_dataset.map(lambda examples: gemma3_messages_to_clm_formatter(examples, tokenizer))
                new_train_dataset = new_train_dataset.select_columns(['text'])
                train_dataset = new_train_dataset

                if test_dataset:
                    test_dataset = messages_form_dataset_filter(test_dataset)
                    test_dataset = test_dataset.map(lambda examples: gemma3_messages_to_clm_formatter(examples, tokenizer))
                print("[INFO] CLM Dataset Ready.")
                print("[INFO]", train_dataset)


            elif len(train_dataset.column_names) == 1:
                train_dataset = rename_dataset_columns(train_dataset, 'text')
                if test_dataset:
                    test_dataset = rename_dataset_columns(test_dataset, 'text')
                print("[INFO] Gemma3 Dataset Columns Renamed. ['text']")

            else:
                train_dataset = rename_dataset_columns(train_dataset, 'system', 'question', 'answer')
                if test_dataset:
                    test_dataset = rename_dataset_columns(test_dataset, 'system', 'question', 'answer')
                print("[INFO] Gemma3 Dataset Columns Renamed. ['system', 'question', 'answer']")

        except:
            raise ValueError("Gemma3 Dataset Columns Error")

        try:
            tokenized_train_dataset = train_dataset.map(
                lambda examples: encode(examples, tokenizer=tokenizer, col_name=train_dataset.column_names[0], tokenizer_max_len=tokenizer_max_len),
                batched=True,
                num_proc=num_proc,
                remove_columns=train_dataset.column_names
            )
            try:
                tokenized_test_dataset = test_dataset.map(
                    lambda examples: encode(examples, tokenizer=tokenizer, col_name=train_dataset.column_names[0], tokenizer_max_len=tokenizer_max_len),
                    batched=True,
                    num_proc=num_proc
                )
            except:
                tokenized_test_dataset = None
        except:
            raise ValueError("[FATAL ERROR] Tokenizing Fail.")

        return tokenized_train_dataset, tokenized_test_dataset

    if selected_trainer_name == 'SFT Trainer' or selected_trainer_name == "KL-SFT Trainer":
        print("[INFO] SFT or KL-SFT Trainer Selected. Return Dataset with Default Processing.")
        try:
            if train_dataset.column_names == ['messages']:
                train_dataset = messages_form_dataset_filter(train_dataset)
                print("[INFO] Gemma3 Dataset Columns Already Renamed. ['messages']")
            else:
                train_dataset = rename_dataset_columns(train_dataset, 'system', 'question', 'answer')
                if test_dataset:
                    test_dataset = rename_dataset_columns(test_dataset, 'system', 'question', 'answer')
                print("[INFO] Gemma3 Dataset Columns Renamed. ['system', 'question', 'answer']")
        except:
            raise ValueError("Gemma3 Dataset Columns Error")

        try:
            if train_dataset.column_names == ['messages']:
                print("[INFO] Gemma3 Dataset Columns Already Renamed. ['messages']")
            else:
                train_dataset = train_dataset.map(lambda examples: gemma_sft_formatter(examples))
                try:
                    test_dataset = test_dataset.map(lambda examples: gemma_sft_formatter(examples))
                except:
                    test_dataset = None
                print("[INFO] Gemma3 Apply Chat Template. Column Name: text")
        except:
            raise ValueError("[FATAL ERROR] gemma formatter Fail.")

        return train_dataset, test_dataset

    elif selected_trainer_name == 'DPO Trainer':
        print("[INFO] DPO Trainer Selected. Return Dataset with Default Processing.")
        if len(train_dataset.column_names) == 3:
            try:
                train_dataset = rename_dataset_columns(train_dataset, 'prompt', 'chosen', 'rejected')
                if test_dataset:
                    test_dataset = rename_dataset_columns(test_dataset, 'prompt', 'chosen', 'rejected')
                print("[INFO] Gemma3 Dataset Columns Renamed. ['prompt', 'chosen', 'rejected']")
            except:
                raise ValueError("Gemma3 Dataset Columns Error")

        elif len(train_dataset.column_names) == 4:
            try:
                train_dataset = rename_dataset_columns(train_dataset, 'system', 'prompt', 'chosen', 'rejected')
                if test_dataset:
                    test_dataset = rename_dataset_columns(test_dataset, 'system', 'prompt', 'chosen', 'rejected')
                print("[INFO] Gemma3 Dataset Columns Renamed. ['system', 'prompt', 'chosen', 'rejected']")
            except:
                raise ValueError("Gemma3 Dataset Columns Error")

        return train_dataset, test_dataset
