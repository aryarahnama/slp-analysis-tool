import argparse
import logging
import os
import time

from parsing.scan import scan_directory
from reports.controller import generate_controller_report
from reports.player import generate_player_report
from parsing.cache import get_save_path, is_save_file, save_scan_results, load_scan_results
from plotting.generate import generate_controller_plot_tasks, generate_player_plot_tasks
from plotting.runner import run_plot_tasks
from utils.misc import clear_logs, format_time


def run(path, aggregate="controller", generate_report=True, make_plots=False, threads=1, save=False, directory_mode=False):
    """Run the replay analysis pipeline for a file or directory.

    Loads previously saved scan results when possible, otherwise scans the
    specified replay file or directory. Optionally saves newly generated
    scan results, generates an aggregate report, and creates plots from the
    collected metrics.

    Args:
        path: Path to a replay file, replay directory, or save file.
        aggregate: Level at which results are aggregated. Supported values
            are "controller" and "player".
        generate_report: Whether to generate an aggregate analysis report.
        make_plots: Whether to generate plots from the collected metrics.
        threads: Number of worker processes to use when scanning replays or
            generating plots.
        save: Whether to save newly generated scan results for later use.
        directory_mode: Whether path refers to a directory when determining
            the save-file path.

    Returns:
        The generated report DataFrame if generate_report is True, otherwise
        None.
    """

    if (not directory_mode and is_save_file(path)):
        (df, stage_counts, char_counts, player_stats, plot_data) = load_scan_results(path)

        print("Successfully loaded save file")
    else:
        (df, stage_counts, char_counts, player_stats, plot_data) = scan_directory(path, max_workers=threads)

        if save:
            try:
                save_data = (df, stage_counts, char_counts, player_stats, plot_data)

                save_path = get_save_path(path, directory_mode)

                save_scan_results(save_path, save_data)

                print(f"Successfully created save file: {save_path}")
            except Exception:
                print(f"Failed to create save file.")

    report_df, tasks = None, None

    if generate_report:
        if aggregate == "controller":
            report_df = generate_controller_report(df, stage_counts, char_counts, player_stats)
        else:
            report_df = generate_player_report(df, stage_counts, char_counts, player_stats)

    if make_plots:
        if aggregate == "controller":
            tasks = generate_controller_plot_tasks(plot_data)
        else:
            tasks = generate_player_plot_tasks(plot_data)

        run_plot_tasks(tasks, threads=threads)

    return report_df


def evaluate_args():
    """Parse and validate command-line arguments for the SLP analysis tool.

    Configures the command-line interface, validates the selected input path,
    aggregation mode, and number of worker threads, and converts the supplied
    arguments into values suitable for the analysis pipeline.

    Returns:
        A tuple containing:
            directory: Directory path supplied through the --dir argument,
                or None if a file was specified.
            file: File path supplied through the --file argument, or None if
                a directory was specified.
            aggregate: Aggregation mode, either "controller" or "player".
            report: Whether report generation was requested.
            plots: Whether plotting output was requested.
            threads: Number of worker threads to use.
            save: Whether parsed scan results should be saved to a cache file.

    Raises:
        ValueError: If the specified input path does not exist, the
            aggregation mode is invalid, or the requested number of threads
            is outside the available range.
    """

    max_threads_cpu = int(os.cpu_count()) if os.cpu_count() else 1

    parser = argparse.ArgumentParser(
        prog="SLP Analysis",
        description="Compute stats and controller types for .slp files",
    )

    input_group = parser.add_mutually_exclusive_group(required=True)

    input_group.add_argument(
        "--dir",
        "-d",
        help="Directory containing .slp or .zip files",
    )

    input_group.add_argument(
        "--file",
        "-f",
        help="Specific .slp or .zip file to process",
    )

    # --- aggregation mode ---
    parser.add_argument(
        "--aggregate",
        "-a",
        choices=["controller", "player"],
        default="controller",
        help="Aggregation method: 'controller' or 'player'",
    )

    # --- reporting flag ---
    parser.add_argument(
        "--report",
        "-r",
        action="store_true",
        help="Enable report generation",
    )

    # --- plotting flag ---
    parser.add_argument(
        "--plots",
        "-p",
        action="store_true",
        help="Enable plotting output",
    )

    # --- backup flag ---
    parser.add_argument(
        "--save",
        "-s",
        action="store_true",
        help="Save parsed scan results to a cache file",
    )

    # --- number of threads ---
    parser.add_argument(
        "--threads",
        "-t",
        default="max",
        help=(f"Number of threads to use (1-{max_threads_cpu}) or 'max'"
              if max_threads_cpu > 1 else f"Only 1 thread available"),
    )

    args = parser.parse_args()

    aggregate = str(args.aggregate).lower()

    if args.dir is not None:
        if not os.path.isdir(str(args.dir)):
            raise ValueError("directory must exist")
    elif args.file is not None:
        if not os.path.isfile(str(args.file)):
            raise ValueError("file must exist")
    else:
        raise ValueError("must provide either a directory or a file")   

    if aggregate != "controller" and aggregate != "player":
        raise ValueError("aggregate must be 'controller' or 'player'")

    if str(args.threads).lower() == "max":
        cpu_count = max_threads_cpu
        threads = cpu_count if (cpu_count is not None and cpu_count > 0) else 1
    else:
        try:
            threads = int(args.threads)
        except ValueError:
            if max_threads_cpu == 1:
                raise ValueError(f"threads must be an integer (1) or 'max'")
            else:
                raise ValueError(
                    f"threads must be an integer (1-{max_threads_cpu}) or 'max'")

        if threads < 1 or threads > max_threads_cpu:
            raise ValueError(
                f"threads must be in range 1-{max_threads_cpu}")
        elif threads != 1 and max_threads_cpu == 1:
            raise ValueError(
                f"only 1 thread available")

    return args.dir, args.file, aggregate, args.report, args.plots, threads, args.save


if __name__ == "__main__":
    dir_path, file_path, aggregate, report, plots, threads, save = evaluate_args()
    path = None

    print("\n--- Arguments ---")

    if dir_path:
        print(f"Directory: {dir_path}")
        path = dir_path
    elif file_path:
        print(f"File: {file_path}")
        path = file_path

    print(f"Aggregate: {aggregate}")
    print(f"Report enabled: {report}")
    print(f"Plots enabled: {plots}")
    print(f"Backup enabled: {save}")
    print(f"Threads: {threads}")
    print("-----------------\n")

    clear_logs()

    start = time.perf_counter()

    run(
        path=path,
        aggregate=aggregate,
        generate_report=report,
        make_plots=plots,
        threads=threads,
        save=save,
        directory_mode=dir_path is not None,
    )

    elapsed = time.perf_counter() - start

    print(f"\nTotal runtime: {format_time(elapsed)} ({elapsed:.2f}s)")
    
    logging.shutdown()
