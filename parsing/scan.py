from collections import Counter, defaultdict, deque
from pathlib import Path
from tqdm.contrib.concurrent import process_map

import gzip
import math
import os
import pandas as pd
import shutil
import tempfile
import zipfile

from config import (TARGET_CHUNK_BYTES, MIN_CHUNKS_PER_WORKER, MAX_CHUNKS_PER_WORKER,
                    MIN_CHUNKSIZE, MAX_CHUNKSIZE, SCANNING_LOG_DIR, SLP_FILE, GZ_SLP_FILE, ZIP_FILE)
from parsing.process import process_slp
from utils.misc import dd_list, ddd_list, create_logger
from utils.tasks import FileTask, ZipTask


parsing_logger = create_logger("ParsingLogger", SCANNING_LOG_DIR)


def classify_file(filename):
    """Determine whether a filename is an SLP or compressed SLP file.

    Args:
        filename: Name of the file to classify.

    Returns:
        A tuple containing:
            is_slp: Whether the filename has the standard SLP extension.
            is_gz_slp: Whether the filename has the gzip-compressed SLP
                extension.
    """

    filename = filename.lower()

    is_slp = filename.endswith(SLP_FILE)
    is_gz_slp = filename.endswith(GZ_SLP_FILE)

    return is_slp, is_gz_slp


def discover_slp_files(root_path):
    """Discover SLP replay files and create processing tasks for them.

    Searches a file or directory recursively for SLP files, gzip-compressed
    SLP files, and ZIP archives containing either type of replay. Each
    discovered replay is represented as either a FileTask or ZipTask.

    Args:
        root_path: Path to a file or directory to search. If a directory is
            provided, all of its subdirectories are searched recursively.

    Returns:
        A list of FileTask and ZipTask objects representing the discovered
        replay files.

    Raises:
        ValueError: If root_path does not exist.
    """

    tasks = []
    root = Path(root_path)

    if root.is_dir():
        paths = root.rglob("*")
    elif root.is_file():
        paths = [root]
    else:
        raise ValueError(f"Path does not exist: {root_path}")

    for path in paths:
        if not path.is_file():
            continue

        filename = path.name.lower()
        suffix = path.suffix.lower()

        is_slp, is_gz_slp = classify_file(filename)

        if is_slp or is_gz_slp:
            tasks.append(
                FileTask(
                    path=str(path),
                    size=path.stat().st_size,
                    compressed=is_gz_slp
                )
            )

        elif suffix == ZIP_FILE:
            try:
                with zipfile.ZipFile(path, "r") as zf:
                    for info in zf.infolist():
                        if info.is_dir():
                            continue

                        is_slp, is_gz_slp = classify_file(info.filename)

                        if is_slp or is_gz_slp:
                            tasks.append(
                                ZipTask(
                                    zip_path=str(path),
                                    member=info.filename,
                                    size=info.file_size,
                                    compressed=is_gz_slp
                                )
                            )
            except zipfile.BadZipFile:
                parsing_logger.error(f"Bad zip file: {path}")

    return tasks


def remove_temp_file(temp_path):
    """Remove a temporary file if it exists.

    Args:
        temp_path: Path to the temporary file to remove.

    Returns:
        True if the file was successfully removed, or False if the path
        does not exist or the file could not be removed.
    """

    if not temp_path or not os.path.exists(temp_path):
        return False

    try:
        os.remove(temp_path)

        return True

    except Exception as e:
        parsing_logger.warning(f"Failed to remove temp file {temp_path}: {e}")

        return False


def stream_to_temp_slp(stream):
    """Write a replay stream to a temporary SLP file.

    Args:
        stream: Binary file-like object containing SLP replay data.

    Returns:
        The path to the newly created temporary SLP file.
    """

    with tempfile.NamedTemporaryFile(
        suffix=SLP_FILE,
        delete=False
    ) as tmp:
        shutil.copyfileobj(stream, tmp)

        return tmp.name


def process_slp_wrapper(task):
    """Process an SLP replay task, decompressing it when necessary.

    Handles regular SLP files, gzip-compressed SLP files, and SLP files
    contained within ZIP archives. Compressed or archived replays are
    extracted to temporary SLP files before being passed to process_slp.
    Temporary files are removed after processing.

    Args:
        task: A FileTask or ZipTask describing the replay to process.

    Returns:
        The analysis results produced by process_slp, or False if the replay
        is filtered out by process_slp. Returns None if an error occurs while
        processing the task.
    """

    temp_path = None

    try:
        match task:
            case FileTask(path, _, compressed):
                if compressed:
                    with gzip.open(path, "rb") as gz_file:
                        temp_path = stream_to_temp_slp(gz_file)

                    return process_slp(temp_path)

                return process_slp(path)

            case ZipTask(zip_path, member, _, compressed):
                with zipfile.ZipFile(zip_path, "r") as zf:
                    with zf.open(member) as zipped_file:

                        if compressed:
                            with gzip.GzipFile(fileobj=zipped_file) as gz_file:
                                temp_path = stream_to_temp_slp(gz_file)
                        else:
                            temp_path = stream_to_temp_slp(zipped_file)

                return process_slp(temp_path)

    except Exception as e:
        parsing_logger.error(f"Error processing {task}: {e}", exc_info=True)

        return None

    finally:
        remove_temp_file(temp_path)


def determine_chunksize_and_order(tasks, n_workers):
    """Determine the processing order and chunk size for replay tasks.

    Tasks are ordered by file size using an alternating largest-to-smallest
    strategy to help balance processing time across workers. The chunk size
    is then selected based on the average task size and constrained by the
    configured worker and chunk-size limits.

    Args:
        tasks: List of replay processing tasks. Each task must provide a
            size attribute representing its file size in bytes.
        n_workers: Number of workers available to process the tasks.

    Returns:
        A tuple containing:
            reordered: The replay tasks reordered to balance the expected
                processing workload.
            chunksize: Number of tasks to assign to each processing chunk.
    """

    if not tasks:
        return [], 1
    elif len(tasks) == 1:
        return tasks, 1

    def determine_chunksize(files, n_workers, target_chunk_bytes=TARGET_CHUNK_BYTES, min_chunks_per_worker=MIN_CHUNKS_PER_WORKER, max_chunks_per_worker=MAX_CHUNKS_PER_WORKER, min_chunksize=MIN_CHUNKSIZE, max_chunksize=MAX_CHUNKSIZE):
        def is_even(n):
            return n % 2 == 0
        
        total_bytes = sum(size for _, size in files)
        avg_bytes = total_bytes / len(files)

        # files per chunk based on average file size
        raw_chunksize = max(1, round(target_chunk_bytes / max(avg_bytes, 1)))

        # Constrain based on worker count
        min_total_chunks = n_workers * min_chunks_per_worker
        max_total_chunks = n_workers * max_chunks_per_worker

        # If chunks are too large or too small, adjust chunksize
        if math.ceil(len(files) / raw_chunksize) < min_total_chunks:
            raw_chunksize = math.ceil(len(files) / min_total_chunks)
        elif math.ceil(len(files) / raw_chunksize) > max_total_chunks:
            raw_chunksize = math.ceil(len(files) / max_total_chunks)

        chunksize = max(min_chunksize, min(raw_chunksize, max_chunksize))

        # Ensure files are taken in "pairs"
        if not is_even(chunksize):
            chunksize -= 1

        return chunksize

    def determine_file_ordering(files):
        # Largest -> smallest, then alternate
        files.sort(key=lambda x: x[1], reverse=True)
        dq = deque(files)
        balanced = []
        take_large = True

        while dq:
            if take_large:
                balanced.append(dq.popleft())
            else:
                balanced.append(dq.pop())

            take_large = not take_large

        reordered = [file for file, _ in balanced]

        return reordered

    files = [(task, task.size) for task in tasks]

    chunksize = determine_chunksize(files, n_workers)

    reordered = determine_file_ordering(files)

    return reordered, chunksize


def merge_scan_results(results):
    """Merge the results produced by parallel replay processing.

    Combines per-game rows, stage and character counts, player statistics,
    and plotting data from each processed replay. Failed and skipped replays
    are tracked separately and are excluded from the merged analysis data.

    Args:
        results: Iterable of results returned by process_slp. Each successful
            result contains per-game rows, stage counts, character counts,
            player statistics, and plotting data. None indicates a failed
            replay, while False indicates a replay that was intentionally
            skipped.

    Returns:
        A tuple containing:
            df: DataFrame containing the combined per-player game results.
            stage_counts: Combined counts of games played on each stage.
            char_counts: Combined counts of players using each character.
            player_stats: Combined per-player controller classification
                counts.
            merged_plot_data: Combined metric data grouped by controller
                classification and player.
            skipped_replays: Number of replays intentionally excluded from
                analysis.
            failed_replays: Number of replays that failed during processing.
    """

    all_rows = []
    stage_counts = Counter()
    char_counts = Counter()
    player_stats = defaultdict(Counter)

    merged_plot_data = {
        "controller": defaultdict(ddd_list),
        "player": defaultdict(dd_list),
    }

    failed_replays = 0
    skipped_replays = 0

    for res in results:
        if not res:
            if res is None:
                failed_replays += 1
            else:
                skipped_replays += 1

            continue

        local_rows, local_stage, local_char, local_player, local_plot = res

        all_rows.extend(local_rows)
        stage_counts.update(local_stage)
        char_counts.update(local_char)

        for k, v in local_player.items():
            player_stats[k].update(v)

        for char, ctrl_data in local_plot["controller"].items():
            for ctrl_type, metrics in ctrl_data.items():
                for metric, values in metrics.items():
                    merged_plot_data["controller"][char][ctrl_type][metric].extend(
                        values)

        for code, metrics in local_plot["player"].items():
            for metric, values in metrics.items():
                merged_plot_data["player"][code][metric].extend(values)

    df = pd.DataFrame(all_rows)

    return (
        df,
        stage_counts,
        char_counts,
        player_stats,
        merged_plot_data,
        skipped_replays,
        failed_replays,
    )


def scan_directory(root_path, max_workers=None, chunksize=None):
    """Scan replay files in a directory using parallel processing.

    Discovers SLP replay files under the specified path, determines an
    appropriate processing order and chunk size, and processes the replays
    across multiple workers. The results from each replay are then merged
    into aggregate analysis data.

    Args:
        root_path: Path to a file or directory containing SLP replay files.
        max_workers: Maximum number of worker processes to use. If None, the
            number of workers defaults to the number of available CPU cores.
        chunksize: Number of replay tasks assigned to each worker at a time.
            If None, the chunk size is determined automatically based on the
            discovered replay files and number of workers.

    Returns:
        A tuple containing:
            df: DataFrame containing the combined per-player game results.
            stage_counts: Counts of games played on each stage.
            char_counts: Counts of players using each character.
            player_stats: Per-player controller classification counts.
            merged_plot_data: Combined metric data grouped by controller
                classification and player.
    """

    if max_workers is None:
        max_workers = os.cpu_count()

    tasks = discover_slp_files(root_path)

    if chunksize is None:
        tasks, chunksize = determine_chunksize_and_order(
            tasks,
            n_workers=max_workers
        )

    print(f"Total number of replays: {len(tasks)}\n")

    results = process_map(
        process_slp_wrapper,
        tasks,
        max_workers=max_workers,
        chunksize=chunksize,
        desc="Scanning replay files"
    )

    parsing_logger.handlers.clear()

    (
        df,
        stage_counts,
        char_counts,
        player_stats,
        merged_plot_data,
        skipped_replays,
        failed_replays,
    ) = merge_scan_results(results)

    print()
    print(f"Number of replays skipped: {skipped_replays}")
    print(f"Number of replays failed: {failed_replays}\n")

    return df, stage_counts, char_counts, player_stats, merged_plot_data
