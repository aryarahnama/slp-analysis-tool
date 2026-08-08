from collections import defaultdict
from multiprocessing_logging import install_mp_handler

import logging
import math
import os
import unicodedata

from config import (MAX_RAW_THRESHOLD, DZ_THRESHOLD, SECONDS_PER_MINUTE, MINUTES_PER_HOUR,
                    HOURS_PER_DAY, EARLY_AERIAL_FRAMES, SCANNING_LOG_DIR, PLOT_LOG_DIR)


def dd_list():
    """Create a defaultdict whose values are empty lists.

    Returns:
        defaultdict: A defaultdict(list) instance that automatically
            initializes missing keys to empty lists.
    """
    return defaultdict(list)


def ddd_list():
    """Create a nested defaultdict whose values are defaultdict(list).

    Returns:
        defaultdict: A two-level defaultdict where missing outer keys
            initialize to defaultdict(list).
    """
    return defaultdict(dd_list)


def ensure_dir(path):
    """Create a directory and any missing parent directories.

    Args:
        path (str | os.PathLike): Path to the directory to create.

    The function does nothing if the directory already exists.
    """

    os.makedirs(path, exist_ok=True)


def format_time(seconds):
    """Format a duration in seconds using human-readable time units.

    Durations are displayed using days, hours, minutes, and seconds,
    omitting leading zero-valued units.

    Args:
        seconds (int | float): Duration in seconds.

    Returns:
        str: Formatted duration, such as ``"45s"``, ``"12m 30s"``,
            ``"2h 15m 4s"``, or ``"1d 3h 20m 10s"``.
    """

    seconds = int(seconds)

    seconds_per_hour = SECONDS_PER_MINUTE * MINUTES_PER_HOUR
    seconds_per_day = seconds_per_hour * HOURS_PER_DAY

    d = seconds // seconds_per_day
    h = (seconds % seconds_per_day) // seconds_per_hour
    m = (seconds % seconds_per_hour) // SECONDS_PER_MINUTE
    s = seconds % SECONDS_PER_MINUTE

    if d > 0:
        return f"{d}d {h}h {m}m {s}s"
    elif h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    else:
        return f"{s}s"


def clear_logs():
    """Clear the configured scanning and plotting log files.

    Existing log files are truncated to zero length. Files that do not
    exist or cannot be opened are skipped without raising an exception.
    """

    LOG_FILES = [SCANNING_LOG_DIR, PLOT_LOG_DIR]

    for log_file in LOG_FILES:
        if os.path.exists(log_file):
            try:
                with open(log_file, "w"):
                    pass
            except Exception:
                continue


def create_logger(name, file_name):
    """Create and configure a multiprocessing-compatible file logger.

    Any existing handlers attached to the logger are removed before
    configuring the new file handler.

    Args:
        name (str): Name of the logger.
        file_name (str | os.PathLike): Path to the log file.

    Returns:
        logging.Logger: Configured logger with INFO-level file logging
            and multiprocessing support.
    """

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # remove any old handlers

    handler = logging.FileHandler(file_name, mode="a")
    formatter = logging.Formatter(
        "%(asctime)s - %(processName)s - %(levelname)s - %(message)s\n")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    install_mp_handler()  # attach multiprocessing handler

    return logger


def normalize_connect_code(code):
    """Normalize a Slippi connect code into a consistent representation.

    Full-width Unicode characters are converted to their ASCII-compatible
    forms, whitespace is removed, and the resulting code is converted to
    uppercase.

    Args:
        code (str | None): Connect code to normalize.

    Returns:
        str | None: Normalized connect code, or None if the input is empty.
    """

    if not code:
        return None

    # Convert full-width → normal ASCII
    code = unicodedata.normalize("NFKC", code)

    # Remove spaces
    code = code.replace(" ", "").strip()

    # Uppercase
    return code.upper()


def process_analog_stick(x, y, deadzone=True):
    """Normalize and process a raw analog stick position.

    The input coordinates are first constrained to the maximum permitted
    magnitude. A component-wise deadzone can then be applied before the
    coordinates are normalized to the range approximately [-1, 1].

    Args:
        x (float): Raw horizontal stick coordinate.
        y (float): Raw vertical stick coordinate.
        deadzone (bool): Whether to set small horizontal and vertical
            components within the configured deadzone to zero.

    Returns:
        tuple[float, float]: Processed and normalized ``(x, y)`` coordinates.
    """

    magnitude = math.sqrt(x**2 + y**2)

    fx = x
    fy = y
    if magnitude > MAX_RAW_THRESHOLD:
        shrinkFactor = MAX_RAW_THRESHOLD / magnitude
        if fx > 0:
            fx = math.floor(fx * shrinkFactor)
            fy = math.floor(fy * shrinkFactor)
        else:
            fx = math.ceil(fx * shrinkFactor)
            fy = math.ceil(fy * shrinkFactor)

    # Deadzone
    if deadzone:
        if abs(fx) < DZ_THRESHOLD:
            fx = 0
        if abs(fy) < DZ_THRESHOLD:
            fy = 0

    fx = round(fx)
    fy = round(fy)
    return (fx / MAX_RAW_THRESHOLD, fy / MAX_RAW_THRESHOLD)


def build_frame_hist(frames_list):
    """Build a histogram of aerial frames within the configured range.

    Each frame value from 1 through ``EARLY_AERIAL_FRAMES`` is counted.
    Values outside this range are ignored.

    Args:
        frames_list (Iterable[int]): Collection of frame numbers to count.

    Returns:
        dict[int, int]: Mapping from frame number to the number of
            occurrences. All frame numbers in the configured range are
            included, even when their count is zero.
    """

    hist = {i: 0 for i in range(1, EARLY_AERIAL_FRAMES + 1)}

    for f in frames_list:
        if 1 <= f <= EARLY_AERIAL_FRAMES:
            hist[f] += 1

    return hist
