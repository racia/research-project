import os


def check_or_create_directory(path: str) -> None:
    """
    Check if the directory exists, if not create it.

    :param path: path to the directory
    """
    if not os.path.exists(path):
        os.makedirs(path)


def is_empty_file(file_path: Path) -> bool:
    """
    Checks if the file exists and is empty.

    :param file_path: the file path to check
    :return: True if file exists and is empty
             False if file is non-empty
    """
    return os.path.isfile(file_path) and os.path.getsize(file_path) == 0
