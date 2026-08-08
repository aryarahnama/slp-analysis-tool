import os

from utils.misc import ensure_dir
from utils.tasks import HistTask, ECDFTask, OverlayTask, ECDFOverlayTask, StatsTask


def generate_controller_plot_tasks(plot_data):
    """Generate plotting tasks for controller metrics by character.

    Creates histogram, ECDF, overlay, and statistical comparison tasks for
    metrics with available Analog and Digital controller data. Tasks are
    organized into character-specific output directories under
    ``plots/characters``.

    Args:
        plot_data: Mapping containing controller metric data grouped by
            character and controller classification.

    Returns:
        A list of plotting tasks for the available controller metrics.
        Metrics without any data are skipped.
    """

    base = "plots/characters"
    ensure_dir(base)

    tasks = []

    for char, controller_data in plot_data["controller"].items():
        char_dir = os.path.join(base, char)
        ensure_dir(char_dir)

        analog = controller_data.get("Analog", {})
        digital = controller_data.get("Digital", {})

        metrics = set(analog.keys()) | set(digital.keys())

        for metric in metrics:
            a_data = analog.get(metric, [])
            d_data = digital.get(metric, [])

            if not a_data and not d_data:
                continue

            # Analog only
            if a_data:
                tasks.append(HistTask(
                    a_data,
                    f"{char} Analog {metric}",
                    os.path.join(char_dir, f"{metric}_analog.png")
                ))

                tasks.append(ECDFTask(
                    a_data,
                    f"{char} Analog {metric} (ECDF)",
                    os.path.join(char_dir, f"{metric}_analog_ecdf.png")
                ))

            # Digital only
            if d_data:
                tasks.append(
                    HistTask(
                        data=d_data,
                        title=f"{char} Digital {metric}",
                        path=os.path.join(char_dir, f"{metric}_digital.png"),
                    )
                )

                tasks.append(ECDFTask(
                    d_data,
                    f"{char} Digital {metric} (ECDF)",
                    os.path.join(char_dir, f"{metric}_digital_ecdf.png")
                ))

            if a_data and d_data:
                # Overlay
                tasks.append(
                    OverlayTask(
                        a=a_data,
                        b=d_data,
                        label_a="Analog",
                        label_b="Digital",
                        title=f"{char} {metric} (Overlay)",
                        path=os.path.join(char_dir, f"{metric}_overlay.png"),
                    )
                )

                tasks.append(
                    ECDFOverlayTask(
                        a=a_data,
                        b=d_data,
                        label_a="Analog",
                        label_b="Digital",
                        title=f"{char} {metric} (Overlay)",
                        path=os.path.join(
                            char_dir, f"{metric}_overlay_ecdf.png"),
                    )
                )

                # Stats
                tasks.append(
                    StatsTask(
                        a=a_data,
                        b=d_data,
                        title=f"{char} {metric} (Stats)",
                        path=os.path.join(char_dir, f"{metric}_stats.png"),
                    )
                )

    return tasks


def generate_player_plot_tasks(plot_data):
    """Generate histogram plotting tasks for player-level metrics.

    Creates one histogram task for each available metric for each player.
    Tasks are organized into player-specific output directories under
    ``plots/players``.

    Args:
        plot_data: Mapping containing metric data grouped by player connect
            code.

    Returns:
        A list of histogram plotting tasks for the available player metrics.
        Metrics without data are skipped.
    """

    base = "plots/players"
    ensure_dir(base)

    tasks = []

    for code, metrics in plot_data["player"].items():
        player_dir = os.path.join(base, code.replace("#", "_"))
        ensure_dir(player_dir)

        for metric, data in metrics.items():
            if not data:
                continue

            tasks.append(
                HistTask(
                    data=data,
                    title=f"{code} {metric}",
                    path=os.path.join(player_dir, f"{metric}.png"),
                )
            )

    return tasks
