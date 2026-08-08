from pathlib import Path

import pickle

from config import SAVE_FILE


def get_save_path(path, is_directory=False):
    """Generate the path used to store scan results.

    If the provided path is a directory, the save file is created inside
    the directory using the directory name as the file name. Otherwise,
    the existing file suffix is replaced with the configured save suffix.

    Args:
        path: Input file or directory path.
        is_directory: Whether `path` represents a directory.

    Returns:
        The generated save file path as a string.
    """
    
    p = Path(path)

    if is_directory:
        return str(p / f"{p.name}{SAVE_FILE}")

    return str(p.with_suffix(SAVE_FILE))


def is_save_file(path):
    """Check whether a path has the configured save file extension.

    Args: 
        path: Path to check.

    Returns: 
        True if the path ends with the configured save file suffix,
        otherwise False.
    """

    return str(path).lower().endswith(SAVE_FILE)


def save_scan_results(path, data):
    """Serialize scan results to a file using pickle. 
    
    Args: 
        path: Destination path for the saved results. 
        data: Scan results to serialize.
    """

    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_scan_results(path):
    """Load serialized scan results from a pickle file. 
    
    Args: 
        path: Path to the saved scan results. 
    
    Returns: 
        The deserialized scan results. 
    """

    with open(path, "rb") as f:
        return pickle.load(f)