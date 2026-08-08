import numpy as np
import pandas as pd


def counts_with_percentages(series):
    """Convert a count series into counts and percentage values.

    Sorts the supplied counts in descending order and calculates the
    percentage represented by each value. If the series has no nonzero
    total, the original series is returned unchanged.

    Args:
        series: Pandas Series containing counts for each category.

    Returns:
        A DataFrame containing the original counts and their percentages,
        or the original Series if its total is zero or undefined.
    """

    series = series.sort_values(ascending=False)
    series_sum = series.sum()
    no_sum = (series_sum is None) or np.isclose(series_sum, 0.0)
    
    if no_sum:
        return series

    return pd.DataFrame({
        "count": series,
        "percent (%)": (series / series_sum * 100).round(2)
    })


def output_eda(df, stage_counts, char_counts):
    """Print high-level exploratory statistics for replay data.

    Reports the total number of unique replays, the number of unique players,
    the date range of the dataset, and the distributions of stages,
    characters, and controller classifications when the corresponding data
    is available.

    Args:
        df: DataFrame containing per-player, per-game replay metrics.
        stage_counts: Counts of games played on each stage.
        char_counts: Counts of players using each character.

    """

    if df is None or df.empty:
        return

    print(f"Total replays scanned: {df['file'].nunique()}")

    df["start_at"] = pd.to_datetime(df["start_at"], errors="coerce")
    
    players = pd.Series(df["connect_code"])
    dates = pd.Series(df["start_at"])
    classes = pd.Series(df["classification"])
    are_all_players_null = players.isnull().all()
    are_all_dates_null = dates.isnull().all()
    are_all_classes_null = classes.isnull().all()

    if not are_all_players_null:
        print(f"Number of unique players: {players.nunique()}")
     
    if not are_all_dates_null:
        min_date, max_date = dates.min(), dates.max()
        
        print(f"Date range: {min_date} → {max_date}")

    print("\n--- Stage distribution ---")
    print(counts_with_percentages(pd.Series(stage_counts)))

    print("\n--- Character distribution ---")
    print(counts_with_percentages(pd.Series(char_counts)))

    if not are_all_classes_null:
        print("\n--- Controller distribution ---")
        print(counts_with_percentages(classes.value_counts()))
