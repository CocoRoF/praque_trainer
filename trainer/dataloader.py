import logging
import datasets
from huggingface_hub import login
from trainer.utils.dataloader_toolkit import DataLoader, is_valid_input
from trainer.utils.constants_loader import get_constant_list

logger = logging.getLogger(__name__)

def load_dataset_auto(name, data_dir, split, method, dataloader, token, bucket_name):
    try:
        if method == "minio":
            dataset = dataloader.minio_dataset_loader(
                dataset_name=name,
                bucket_name=bucket_name,
                local_save_directory="/datasets/minio",
                save_only=False
            )
            return dataset[split] if split in dataset else dataset
        elif method == "huggingface":
            try:
                return datasets.load_dataset(name, data_dir=data_dir, split=split, token=token)
            except Exception:
                dataset = datasets.load_dataset(name, data_dir=data_dir, token=token)
                return dataset[split] if split in dataset else dataset
        else:
            raise ValueError(f"[FATAL ERROR] Invalid dataset_load_method: {method}")
    except Exception as e:
        raise RuntimeError(f"[FATAL ERROR] Failed to load dataset `{name}` from {method}: {e}")

def is_valid_example(example):
    return all(is_valid_input(example, col_name=col) for col in example.keys())

def select_columns_if_specified(dataset, col_list):
    if col_list:
        try:
            return dataset.select_columns(col_list)
        except Exception as e:
            logger.warning(f"[WARN] Failed to select columns {col_list}: {e}")
    return dataset

def serve_dataset(
    train_data,
    train_data_dir,
    train_data_split,
    test_data,
    test_data_dir,
    test_data_split,
    train_test_split_ratio,
    dataset_main_column=None,
    dataset_sub_column=None,
    dataset_minor_column=None,
    dataset_last_column=None,
    data_filtering=True,
    data_args=None
):
    # login to Hugging Face if needed
    HUGGING_FACE_TOKEN = data_args.hugging_face_token
    if HUGGING_FACE_TOKEN:
        login(token=HUGGING_FACE_TOKEN)

    dataloader = DataLoader(
        data_args.hugging_face_user_id,
        data_args.hugging_face_token,
        data_args.minio_url,
        data_args.minio_access_key,
        data_args.minio_secret_key,
    )

    # Load train dataset
    train_dataset = load_dataset_auto(
        name=train_data,
        data_dir=train_data_dir,
        split=train_data_split,
        method=data_args.dataset_load_method,
        dataloader=dataloader,
        token=HUGGING_FACE_TOKEN,
        bucket_name=data_args.minio_data_load_bucket,
    )

    # Load or split test dataset
    if isinstance(test_data, str) and test_data.strip():
        try:
            test_dataset = load_dataset_auto(
                name=test_data,
                data_dir=test_data_dir,
                split=test_data_split,
                method=data_args.dataset_load_method,
                dataloader=dataloader,
                token=HUGGING_FACE_TOKEN,
                bucket_name=data_args.minio_data_load_bucket,
            )
        except Exception as e:
            logger.warning(f"[WARN] Failed to load test dataset: {e}")
            test_dataset = None
    elif train_test_split_ratio:
        logger.info(f"[INFO] No test dataset found. Splitting train dataset. Ratio = {train_test_split_ratio}")
        split_dataset = train_dataset.train_test_split(test_size=train_test_split_ratio, seed=42)
        train_dataset = split_dataset["train"]
        test_dataset = split_dataset["test"]
    else:
        test_dataset = None

    # Data filtering
    if data_filtering:
        logger.info(f"[INFO][DATASET] Filtering examples with invalid types")
        train_dataset = train_dataset.filter(is_valid_example)
        if test_dataset:
            test_dataset = test_dataset.filter(is_valid_example)

    # Column selection
    col_names = list(filter(None, [dataset_main_column, dataset_sub_column, dataset_minor_column, dataset_last_column]))
    logger.info(f"[INFO][DATASET] Input Columns = {col_names}")

    train_dataset = select_columns_if_specified(train_dataset, col_names)
    if test_dataset:
        test_dataset = select_columns_if_specified(test_dataset, col_names)

    return train_dataset, test_dataset
