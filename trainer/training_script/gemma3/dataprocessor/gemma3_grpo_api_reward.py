from trainer.utils.dataloader_toolkit import rename_dataset_columns

def processor(
    train_dataset,
    test_dataset,
    tokenizer,
    tokenizer_max_len,
    data_args,
):
    """
    Process the dataset based on the task type.

    Args:
        train_dataset: The training dataset.
        test_dataset: The testing dataset.
        tokenizer: The tokenizer to use for processing.
        tokenizer_max_len: The maximum length for tokenization.
        num_proc: The number of processes to use for tokenization.
        data_args: Data arguments containing dataset configuration.

    Returns:
        Processed training and testing datasets.
    """

    # Check if 'prompt' column exists in the datasets
    train_columns = train_dataset.column_names if train_dataset is not None else []
    test_columns = test_dataset.column_names if test_dataset is not None else []

    # If 'prompt' column doesn't exist and dataset_main_column is specified
    if 'prompt' not in train_columns and hasattr(data_args, 'dataset_main_column'):
        main_column = data_args.dataset_main_column

        # Rename main column to 'prompt' in training dataset
        if train_dataset is not None and main_column in train_columns:
            train_dataset = train_dataset.rename_column(main_column, 'prompt')
            train_columns = train_dataset.column_names  # Update column list

        # Rename main column to 'prompt' in test dataset
        if test_dataset is not None and main_column in test_columns:
            test_dataset = test_dataset.rename_column(main_column, 'prompt')
            test_columns = test_dataset.column_names  # Update column list

    # Check if 'ground_truth' column exists, if not rename dataset_sub_column to 'ground_truth'
    if hasattr(data_args, 'dataset_sub_column') and data_args.dataset_sub_column:
        sub_column = data_args.dataset_sub_column

        # Handle training dataset
        if train_dataset is not None:
            train_columns = train_dataset.column_names
            print(f"[DEBUG] Training dataset columns: {train_columns}")
            if 'ground_truth' not in train_columns and sub_column in train_columns:
                train_dataset = train_dataset.rename_column(sub_column, 'ground_truth')
                print(f"[INFO] Renamed column '{sub_column}' to 'ground_truth' in training dataset")
            elif 'ground_truth' in train_columns:
                print(f"[INFO] 'ground_truth' column already exists in training dataset")

        # Handle test dataset
        if test_dataset is not None:
            test_columns = test_dataset.column_names
            print(f"[DEBUG] Test dataset columns: {test_columns}")
            if 'ground_truth' not in test_columns and sub_column in test_columns:
                test_dataset = test_dataset.rename_column(sub_column, 'ground_truth')
                print(f"[INFO] Renamed column '{sub_column}' to 'ground_truth' in test dataset")
            elif 'ground_truth' in test_columns:
                print(f"[INFO] 'ground_truth' column already exists in test dataset")

    return train_dataset, test_dataset
