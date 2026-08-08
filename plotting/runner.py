from tqdm.contrib.concurrent import process_map

from config import PLOT_LOG_DIR
from plotting.plots import plot_hist, plot_overlay, plot_ecdf, plot_ecdf_overlay, plot_stats_summary
from stats.tests import compute_distribution_stats
from utils.misc import create_logger
from utils.tasks import HistTask, ECDFTask, OverlayTask, ECDFOverlayTask, StatsTask


plots_logger = create_logger("PlotsLogger", PLOT_LOG_DIR)


def task_weight(task):
    """Calculate an estimated processing weight for a plotting task.

    The weight is based on the amount of data involved in the task and is
    used to order plotting tasks so that more expensive tasks are processed
    first.

    Args:
        task: A plotting task containing one or more data collections.

    Returns:
        An estimated processing weight based on the task's input data size.
    """

    match task:
        case StatsTask(a, b, _, _):
            return len(a) * len(b)

        case OverlayTask(a, b, _, _, _, _):
            return len(a) + len(b)

        case ECDFOverlayTask(a, b, _, _, _, _):
            return len(a) + len(b)

        case HistTask(data, _, _):
            return len(data)

        case ECDFTask(data, _, _):
            return len(data)

    return 1


def run_plot_task(task):
    """Execute a plotting task based on its task type.

    Dispatches the task to the appropriate plotting or statistical function
    based on the task's type. Statistical tasks first compute distribution
    statistics before generating their summary plot.

    Args:
        task: A plotting task specifying the data, plot configuration, and
            output path.
    """

    match task:
        case HistTask(data, title, path):
            plot_hist(data, title, path)

        case OverlayTask(a, b, label_a, label_b, title, path):
            plot_overlay(a, b, label_a, label_b, title, path)

        case ECDFTask(data, title, path):
            plot_ecdf(data, title, path)

        case ECDFOverlayTask(a, b, label_a, label_b, title, path):
            plot_ecdf_overlay(a, b, label_a, label_b, title, path)

        case StatsTask(a, b, title, path):
            stats = compute_distribution_stats(a, b)
            plot_stats_summary(stats, title, path)


def run_plot_task_wrapper(task):
    """Execute a plotting task while handling processing errors.

    Wraps run_plot_task with exception handling so that a failed task does
    not terminate the processing of the remaining plotting tasks.

    Args:
        task: A plotting task to execute.

    Returns:
        True if the task was successfully processed, otherwise False.
    """

    try:
        run_plot_task(task)

        return True
    except Exception as e:
        plots_logger.error(f"Error processing task {task.path if hasattr(task, "path") else task}: {e}", exc_info=True)
        
        return False


def run_plot_tasks(tasks, threads=1):
    """Execute plotting tasks in parallel.

    Sorts tasks by estimated processing cost using longest-processing-time-
    first scheduling, then distributes the tasks across the specified number
    of worker processes. Reports the number of successfully generated and
    failed plots after processing is complete.

    Args:
        tasks: List of plotting tasks to execute.
        threads: Number of worker processes to use.

    """

    if not tasks:
        return

    # Longest-processing-time-first scheduling
    tasks = sorted(tasks, key=task_weight, reverse=True)

    results = process_map(
        run_plot_task_wrapper,
        tasks,
        max_workers=threads,
        chunksize=1,
        desc="Generating plots",
    )

    plots_logger.handlers.clear()

    successes = sum(list(results))
    failures = len(results) - successes

    print()
    print(f"Plots successfully generated: {successes}")
    print(f"Plots failed: {failures}")
