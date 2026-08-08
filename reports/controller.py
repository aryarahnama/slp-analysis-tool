import pandas as pd

from reports.eda import output_eda


def generate_controller_report(df: pd.DataFrame, stage_counts, char_counts, player_stats):
    """Generate a controller-level statistical report from replay data.

    Produces exploratory data analysis and aggregates movement, ledge and
    intangibility, SDI, and early aerial timing statistics by character and
    controller classification. The resulting statistics are printed by
    category and returned as a DataFrame for further analysis.

    Args:
        df: DataFrame containing per-player, per-game replay metrics.
        stage_counts: Counts of games played on each stage.
        char_counts: Counts of players using each character.
        player_stats: Mapping of player connect codes to counts of controller
            classifications.

    Returns:
        A DataFrame containing the aggregated statistics for each character
        and controller classification. Returns None if df is None or empty.
    """

    if df is None or df.empty:
        return

    print("\n================ CONTROLLER REPORT ================\n")

    output_eda(df, stage_counts, char_counts)

    aerial_cols = []
    for move in ["nair", "fair", "bair", "uair", "dair"]:
        for i in range(1, 7):
            aerial_cols.append(f"{move}_{i}")

    base = (
        df.groupby(["character", "classification"])
        .agg(
            games=("file", "count"),

            # ---------------- DASH DANCE ----------------
            avg_dash_dances=("dash_dances", "mean"),
            max_dash_dances=("dash_dances", "max"),

            total_dash_dance_lengths=("dash_dance_lengths_sum", "sum"),
            total_dash_dance_length_count=("dash_dance_lengths_count", "sum"),

            total_dash_dance_distances=("dash_dance_distances_sum", "sum"),
            total_dash_dance_distances_count=(
                "dash_dance_distances_count", "sum"),

            # ---------------- WAVEDASH ----------------
            avg_wavedashes=("wavedashes", "mean"),
            max_wavedashes=("wavedashes", "max"),

            avg_wavelands=("wavelands", "mean"),
            max_wavelands=("wavelands", "max"),

            total_wavedash_frame=("wavedash_frame_sum", "sum"),
            total_wavedash_frame_count=("wavedash_frame_count", "sum"),

            total_wavedash_angle=("wavedash_angle_sum", "sum"),
            total_wavedash_count=("wavedash_angle_count", "sum"),

            # ---------------- LEDGEDASH / GALINT ----------------
            total_ledgedashes=("intangible_ledgedashes", "sum"),
            total_no_impact_lands=("no_impact_lands", "sum"),
            total_galint=("total_galint", "sum"),
            max_galint=("max_galint", "max"),

            total_ledgedash_angle=("ledgedash_angle_sum", "sum"),
            total_ledgedash_count=("ledgedash_angle_count", "sum"),

            # ---------------- SDI ----------------
            total_sdi_moves=("sdi_moves", "sum"),
            total_sdi_opportunities=("sdi_opportunities", "sum"),

            total_sdi_inputs=("sdi_inputs_sum", "sum"),
            total_sdi_inputs_count=("sdi_inputs_count", "sum"),

            total_sdi_magnitudes=("sdi_magnitudes_sum", "sum"),
            total_sdi_magnitudes_count=("sdi_magnitudes_count", "sum"),

            # ---------------- EARLY AERIALS ----------------
            **{col: (col, "sum") for col in aerial_cols},
        )
        .reset_index()
    )

    base["mean_dash_dance_length"] = (
        base["total_dash_dance_lengths"] /
        base["total_dash_dance_length_count"]
    ).fillna(0)

    base["mean_dash_dance_distance"] = (
        base["total_dash_dance_distances"] /
        base["total_dash_dance_distances_count"]
    ).fillna(0)

    base["mean_wavedash_frame"] = (
        base["total_wavedash_frame"] /
        base["total_wavedash_frame_count"]
    ).fillna(0)

    base["mean_wavedash_angle"] = (
        base["total_wavedash_angle"] /
        base["total_wavedash_count"]
    ).fillna(0)

    base["mean_ledgedash_angle"] = (
        base["total_ledgedash_angle"] /
        base["total_ledgedash_count"]
    ).fillna(0)

    base["mean_galint"] = (
        base["total_galint"] /
        (base["total_ledgedashes"] + base["total_no_impact_lands"])
    ).replace([float("inf")], 0).fillna(0)

    base["mean_sdi_proportion"] = (
        base["total_sdi_moves"] /
        base["total_sdi_opportunities"]
    ).fillna(0)

    base["mean_sdi_inputs"] = (
        base["total_sdi_inputs"] /
        base["total_sdi_inputs_count"]
    ).fillna(0)

    base["mean_sdi_magnitude"] = (
        base["total_sdi_magnitudes"] /
        base["total_sdi_magnitudes_count"]
    ).fillna(0)

    base = base.drop(columns=[
        "total_dash_dance_lengths",
        "total_dash_dance_length_count",
        "total_dash_dance_distances",
        "total_dash_dance_distances_count",
        "total_wavedash_frame",
        "total_wavedash_frame_count",
        "total_wavedash_angle",
        "total_wavedash_count",
        "total_ledgedash_angle",
        "total_ledgedash_count",
        "total_sdi_inputs_count",
        "total_sdi_magnitudes_count",
    ])

    print("\n================ DASH DANCE =================")
    print(
        base.sort_values("avg_dash_dances", ascending=False)
        [["character", "classification", "avg_dash_dances",
            "mean_dash_dance_length", "mean_dash_dance_distance"]]
        .to_string(index=False)
    )

    print("\n================ WAVEDASH / WAVELAND =================")
    print(
        base.sort_values("avg_wavedashes", ascending=False)
        [["character", "classification", "avg_wavedashes",
            "mean_wavedash_frame", "mean_wavedash_angle"]]
        .to_string(index=False)
    )

    print("\n================ LEDGE / GALINT =================")
    print(
        base.sort_values("mean_galint", ascending=False)
        [["character", "classification", "total_ledgedashes", "total_no_impact_lands",
            "mean_galint", "mean_ledgedash_angle"]]
        .to_string(index=False)
    )

    print("\n================ SDI =================")
    print(
        base.sort_values("mean_sdi_inputs", ascending=False)
        [["character", "classification", "total_sdi_moves", "total_sdi_opportunities",
            "mean_sdi_proportion", "mean_sdi_inputs", "mean_sdi_magnitude"]]
        .to_string(index=False)
    )

    print("\n================ EARLY AERIAL TIMING =================")

    for move in ["nair", "fair", "bair", "uair", "dair"]:
        cols = [f"{move}_{i}" for i in range(1, 7)]

        print(f"\n--- {move.upper()} ---")
        print(
            base[["character", "classification"] + cols]
            .sort_values(cols[0], ascending=False)
            .to_string(index=False)
        )
    print()

    return base
