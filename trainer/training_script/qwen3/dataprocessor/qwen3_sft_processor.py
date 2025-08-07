from trainer.utils.dataloader_toolkit import messages_form_dataset_filter, rename_dataset_columns

def convert_two_columns_to_messages(examples):
    """
    Convert 2 columns (user, assistant) to messages format.
    """
    col_names = list(examples.keys())
    user_content = examples[col_names[0]]  # 첫 번째 컬럼: user
    assistant_content = examples[col_names[1]]  # 두 번째 컬럼: assistant

    if user_content is None or assistant_content is None:
        raise ValueError("Value Error: user and assistant content cannot be None")

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content}
    ]

    return {"messages": messages}

def convert_three_columns_to_messages(examples):
    """
    Convert 3 columns (system, user, assistant) to messages format.
    """
    col_names = list(examples.keys())
    system_content = examples[col_names[0]]  # 첫 번째 컬럼: system
    user_content = examples[col_names[1]]     # 두 번째 컬럼: user
    assistant_content = examples[col_names[2]] # 세 번째 컬럼: assistant

    if system_content is None or len(system_content) < 1:
        system_content = "You are a helpful assistant."

    if user_content is None or assistant_content is None:
        raise ValueError("Value Error: user and assistant content cannot be None")

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content}
    ]

    return {"messages": messages}

def processor(
    train_dataset,
    test_dataset,
    tokenizer,
    tokenizer_max_len,
    data_args,
):
    """
    Process the dataset to convert to messages format for SFT training.

    Args:
        train_dataset: The training dataset.
        test_dataset: The testing dataset.
        tokenizer: The tokenizer to use for processing.
        tokenizer_max_len: The maximum length for tokenization.
        data_args: The number of processes to use for processing.

    Returns:
        Processed training and testing datasets with messages format.
    """

    num_columns = len(train_dataset.column_names)
    print(f"[INFO] Dataset has {num_columns} columns: {train_dataset.column_names}")

    try:
        if num_columns == 1:
            # 1개 컬럼: messages 형태라고 가정
            current_col_name = train_dataset.column_names[0]

            if current_col_name != 'messages':
                # 컬럼 이름이 messages가 아니면 변경
                train_dataset = rename_dataset_columns(train_dataset, 'messages')
                if test_dataset:
                    test_dataset = rename_dataset_columns(test_dataset, 'messages')
                print(f"[INFO] Renamed column '{current_col_name}' to 'messages'")

            # messages_form_dataset_filter 적용
            train_dataset = messages_form_dataset_filter(train_dataset)
            if test_dataset:
                test_dataset = messages_form_dataset_filter(test_dataset)
            print("[INFO] Applied messages_form_dataset_filter")

        elif num_columns == 2:
            # 2개 컬럼: user, assistant로 변환
            train_dataset = train_dataset.map(
                convert_two_columns_to_messages,
                remove_columns=train_dataset.column_names
            )
            if test_dataset:
                test_dataset = test_dataset.map(
                    convert_two_columns_to_messages,
                    remove_columns=test_dataset.column_names
                )
            print("[INFO] Converted 2 columns to messages format (user, assistant)")

        elif num_columns == 3:
            # 3개 컬럼: system, user, assistant로 변환
            train_dataset = train_dataset.map(
                convert_three_columns_to_messages,
                remove_columns=train_dataset.column_names
            )
            if test_dataset:
                test_dataset = test_dataset.map(
                    convert_three_columns_to_messages,
                    remove_columns=test_dataset.column_names
                )
            print("[INFO] Converted 3 columns to messages format (system, user, assistant)")

        else:
            raise ValueError(f"Unsupported number of columns: {num_columns}. Expected 1, 2, or 3 columns.")

    except Exception as e:
        raise ValueError(f"[FATAL ERROR] Dataset conversion failed: {e}")

    print(f"[INFO] Final dataset format - Train columns: {train_dataset.column_names}")
    if test_dataset:
        print(f"[INFO] Final dataset format - Test columns: {test_dataset.column_names}")

    print(f"[INFO] Train dataset size: {len(train_dataset)}")
    if test_dataset:
        print(f"[INFO] Test dataset size: {len(test_dataset)}")

    return train_dataset, test_dataset
