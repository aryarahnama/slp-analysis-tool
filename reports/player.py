import pandas as pd

from reports.eda import output_eda


def generate_player_report(df: pd.DataFrame, stage_counts, char_counts, player_stats):
    """Generate a player-level statistical report from replay data.

    Produces exploratory data analysis, controller classification counts,
    movement statistics, ledge and intangibility statistics, SDI statistics,
    and early aerial timing statistics aggregated by player connect code.

    Args:
        df: DataFrame containing per-player, per-game replay metrics.
        stage_counts: Counts of games played on each stage.
        char_counts: Counts of players using each character.
        player_stats: Mapping of player connect codes to counts of controller
            classifications.

    Returns:
        A DataFrame containing the aggregated player-level statistics. Returns
        None if df is None or empty.
    """

    if df is None or df.empty:
        return

    print("\n================ PLAYER REPORT ================\n")

    output_eda(df, stage_counts, char_counts)

    print("\n--- Player classification counts ---")

    player_df = []

    for code, counts in player_stats.items():
        player_df.append({
            "connect_code": code,
            "digital": counts.get("Digital", 0),
            "possibly_digital": counts.get("Possibly Digital", 0),
            "analog": counts.get("Analog", 0),
            "total": sum(counts.values())
        })

    player_df = pd.DataFrame(player_df)
    player_df = player_df.sort_values("total", ascending=False)

    print(player_df.to_string(index=False))

    dash_stats = (
        df.groupby("connect_code")
        .agg(
            avg_dash_dances=("dash_dances", "mean"),
            median_dash_dances=("dash_dances", "median"),
            max_dash_dances=("dash_dances", "max"),
            games=("dash_dances", "count"),

            total_dash_dance_lengths=("dash_dance_lengths_sum", "sum"),
            total_dash_dance_length_count=("dash_dance_lengths_count", "sum"),

            total_dash_dance_distances=("dash_dance_distances_sum", "sum"),
            total_dash_dance_distances_count=(
                "dash_dance_distances_count", "sum")
        )
        .reset_index()
    )

    dash_stats["mean_dash_dance_length"] = (
        dash_stats["total_dash_dance_lengths"] /
        dash_stats["total_dash_dance_length_count"]
    ).fillna(0)

    dash_stats["mean_dash_dance_distance"] = (
        dash_stats["total_dash_dance_distances"] /
        dash_stats["total_dash_dance_distances_count"]
    ).fillna(0)

    dash_stats = dash_stats.drop(
        columns=["total_dash_dance_lengths", "total_dash_dance_length_count",
                 "total_dash_dance_distances", "total_dash_dance_distances_count"]
    )

    print("\n--- Dash dance stats per player ---")
    print(dash_stats.sort_values("avg_dash_dances",
          ascending=False).to_string(index=False))

    dash_stats = dash_stats.drop(
        columns=["games"]
    )

    player_df = player_df.merge(dash_stats, on="connect_code", how="left")

    movement_stats = (
        df.groupby("connect_code")
        .agg(
            avg_wavedashes=("wavedashes", "mean"),
            median_wavedashes=("wavedashes", "median"),
            max_wavedashes=("wavedashes", "max"),

            avg_wavelands=("wavelands", "mean"),
            median_wavelands=("wavelands", "median"),
            max_wavelands=("wavelands", "max"),

            total_wavedash_frame=("wavedash_frame_sum", "sum"),
            total_wavedash_frame_count=("wavedash_frame_count", "sum"),

            total_wavedash_angle=("wavedash_angle_sum", "sum"),
            total_wavedash_count=("wavedash_angle_count", "sum"),

            games=("wavedashes", "count")
        )
        .reset_index()
    )

    movement_stats["mean_wavedash_angle"] = (
        movement_stats["total_wavedash_angle"] /
        movement_stats["total_wavedash_count"]
    ).fillna(0)

    movement_stats["mean_wavedash_frame"] = (
        movement_stats["total_wavedash_frame"] /
        movement_stats["total_wavedash_frame_count"]
    ).fillna(0)

    movement_stats = movement_stats.drop(
        columns=["total_wavedash_frame", "total_wavedash_frame_count",
                 "total_wavedash_angle", "total_wavedash_count"]
    )

    print("\n--- Wavedash / Waveland stats per player ---")
    print(movement_stats.sort_values("mean_wavedash_frame",
          ascending=True).to_string(index=False))

    movement_stats = movement_stats.drop(
        columns=["games"]
    )

    player_df = player_df.merge(movement_stats, on="connect_code", how="left")

    ledgedash_stats = (
        df.groupby("connect_code")
        .agg(
            total_ledgedashes=("intangible_ledgedashes", "sum"),
            total_no_impact_lands=("no_impact_lands", "sum"),
            total_galint=("total_galint", "sum"),

            avg_ledgedashes=("intangible_ledgedashes", "mean"),
            median_ledgedashes=("intangible_ledgedashes", "median"),
            max_ledgedashes=("intangible_ledgedashes", "max"),

            max_galint=("max_galint", "max"),

            total_ledgedash_angle=("ledgedash_angle_sum", "sum"),
            total_ledgedash_count=("ledgedash_angle_count", "sum"),

            games=("intangible_ledgedashes", "count")
        )
        .reset_index()
    )

    ledgedash_stats["mean_ledgedash_angle"] = (
        ledgedash_stats["total_ledgedash_angle"] /
        ledgedash_stats["total_ledgedash_count"]
    ).fillna(0)

    ledgedash_stats["mean_galint"] = (
        ledgedash_stats["total_galint"] /
        (ledgedash_stats["total_ledgedashes"] +
         ledgedash_stats["total_no_impact_lands"])
    ).fillna(0)

    ledgedash_stats = ledgedash_stats.drop(
        columns=["total_ledgedash_angle", "total_ledgedash_count"]
    )

    print("\n--- Ledge / Intangibility stats per player ---")
    print(
        ledgedash_stats
        .sort_values("mean_galint", ascending=False)
        .to_string(index=False)
    )

    ledgedash_stats = ledgedash_stats.drop(
        columns=["games"]
    )

    player_df = player_df.merge(ledgedash_stats, on="connect_code", how="left")

    sdi_stats = (
        df.groupby("connect_code")
        .agg(
            total_sdi_inputs=("sdi_inputs_sum", "sum"),
            total_sdi_inputs_count=("sdi_inputs_count", "sum"),

            total_sdi_magnitudes=("sdi_magnitudes_sum", "sum"),
            total_sdi_magnitudes_count=("sdi_magnitudes_count", "sum"),

            total_sdi_moves=("sdi_moves", "sum"),
            total_sdi_opportunities=("sdi_opportunities", "sum"),

            games=("sdi_inputs_count", "count"),
        )
        .reset_index()
    )

    sdi_stats["mean_sdi_inputs"] = (
        sdi_stats["total_sdi_inputs"] / sdi_stats["total_sdi_inputs_count"]
    ).fillna(0)

    sdi_stats["mean_sdi_magnitude"] = (
        sdi_stats["total_sdi_magnitudes"] /
        sdi_stats["total_sdi_magnitudes_count"]
    ).fillna(0)

    sdi_stats["mean_sdi_proportion"] = (
        sdi_stats["total_sdi_moves"] / sdi_stats["total_sdi_opportunities"]
    ).fillna(0)

    sdi_stats = sdi_stats.drop(
        columns=["total_sdi_inputs_count",
                 "total_sdi_magnitudes", "total_sdi_magnitudes_count"]
    )

    print("\n--- SDI stats per player ---")
    print(sdi_stats.sort_values("mean_sdi_inputs",
          ascending=False).to_string(index=False))

    sdi_stats = sdi_stats.drop(
        columns=["games"]
    )

    player_df = player_df.merge(sdi_stats, on="connect_code", how="left")

    # ----------------------------
    # AERIAL TIMING STATS
    # ----------------------------
    aerial_cols = []
    for move in ["nair", "fair", "bair", "uair", "dair"]:
        for i in range(1, 7):
            aerial_cols.append(f"{move}_{i}")

    aerial_stats = (
        df.groupby("connect_code")[aerial_cols]
        .sum()
        .reset_index()
    )

    print("\n--- Early Aerial timing (counts per frame) ---")
    print(aerial_stats.to_string(index=False))
    print()

    # merge into player_df
    player_df = player_df.merge(aerial_stats, on="connect_code", how="left")

    return player_df
