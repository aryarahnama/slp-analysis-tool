# SLP Analysis

A modular Python application for analyzing **Super Smash Bros. Melee Slippi replay (`.slp`) files**. The project recursively scans replay collections, extracts gameplay and controller-input data, classifies controllers as analog or digital, computes gameplay metrics, performs statistical analysis, and generates reports and visualizations.

The application was designed with a focus on **modularity, scalability, and reproducibility**. Scanning, reporting, statistical analysis, and plotting are separated so that expensive replay processing does not need to be repeated when only the presentation or analysis of the results changes.

## Features

- **Recursive SLP scanning**
  - Recursively searches directories for `.slp` replay files.
  - Supports replay collections organized across nested directories.
  - Can process an individual replay file directly.

- **Compressed replay support**
  - Processes standard `.slp` files.
  - Processes GZIP-compressed Slippi replay files `.slp.gz`.
  - Supports `.zip` archives containing replay files (`.slp` or `.slp.gz`).
  - Supports compressed replay formats without requiring the user to manually extract them.

- **Gameplay metric extraction**
  - Extracts frame-level gameplay information from Slippi replays.
  - Computes metrics related to movement, defense, and gameplay execution.
  - Examples include:
    - Dash dances
    - Wavedashes
    - Wavelands
    - Ledgedashes
    - GALINT
    - SDI
    - Early aerial timing
    - Other frame-level gameplay measurements

- **Controller classification**
  - Analyzes analog-stick input distributions to classify controllers as:
    - `Analog`
    - `Digital`
    - `Possibly Digital`
  - Uses characteristics such as unique coordinates, repeated input holds, angular entropy, off-axis inputs, and stick-rim behavior.

- **Player and controller aggregation**
  - Results can be aggregated by controller type or individual player.
  - Controller aggregation allows comparisons between analog and digital controllers.
  - Player aggregation allows analysis of individual players across their available replays.

- **Statistical analysis**
  - Supports statistical comparison of distributions.
  - Includes:
    - Kolmogorov-Smirnov tests
    - Mann-Whitney U tests
    - Cliff's delta
    - Hodges-Lehmann estimator
    - IQR differences
    - Skewness differences
    - Bootstrap confidence intervals
  - Statistical calculations include optimizations for large datasets to avoid unnecessarily expensive Cartesian-product calculations.

- **Data visualization**
  - Generates:
    - Histograms
    - Empirical cumulative distribution functions (ECDFs)
    - Distribution overlays
    - ECDF comparisons
    - Statistical summary plots
  - ECDF comparisons include visualization of the Kolmogorov-Smirnov statistic.

- **Modular workflow**
  - Scanning, saving, reporting, and plotting are independent operations.
  - Scan results can be saved to a cache file and reused later.
  - Reports and plots can therefore be regenerated without rescanning the original replay collection.

- **Parallel processing**
  - Replay scanning is designed to take advantage of multiple CPU threads/processes.
  - Plot generation also supports parallel task execution.
  - Work is dynamically scheduled to help balance tasks with substantially different processing costs.

- **Progress monitoring and logging**
  - Long-running scanning and plotting operations provide progress information.
  - Separate logging is used for scanning and plotting operations.
  - Multiprocessing-aware logging allows errors from worker processes to be recorded.

- **Data exploration and reporting**
  - Generates summary information about the scanned dataset, including:
    - Number of replays
    - Number of unique players
    - Replay date range
    - Stage distribution
    - Character distribution
    - Controller distribution
  - Aggregated gameplay metrics can be displayed by character and controller classification.

## Project Design

The project is intentionally divided into independent components rather than implementing replay scanning, statistical analysis, and visualization as one monolithic process.

A typical workflow is:

```text
SLP / ZIP files
      │
      ▼
┌───────────────┐
│ Replay Scanner│
└───────┬───────┘
        │
        ▼
┌──────────────────┐
│ Parsed Scan Data │
└───────┬──────────┘
        │
        ├──────────────► Save / Load Cache
        │
        ├──────────────► Console Report
        │
        └──────────────► Statistical Analysis
                               │
                               ▼
                         Visualizations
```

This separation is particularly useful when working with large replay datasets. Replay parsing can be computationally expensive, while generating a report or changing a plot generally does not require parsing the original replay files again.

For example, a dataset can be scanned once and saved:

```bash
python main.py --dir ./replays --save
```

The saved results can then be used for subsequent analysis without repeating the scan.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/aryarahnama/slp-analysis-tool.git
cd slp-analysis-tool
```

### 2. Create a Python virtual environment

It is recommended to use a virtual environment to isolate the project's dependencies.

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

Once the virtual environment is activated:

```bash
pip install -r requirements.txt
```

## Usage

The main program is invoked through `main.py`.

```bash
python main.py [options]
```

Either a directory or an individual file must be specified.

### Input

#### Process a directory

```bash
python main.py --dir ./replays
```

or:

```bash
python main.py -d ./replays
```

The directory is searched recursively for supported replay files.

#### Process a single file

```bash
python main.py --file ./replays/game.slp
```

or:

```bash
python main.py -f ./replays/game.slp
```

ZIP archives containing replay files can also be supplied.

### Aggregation

The `--aggregate` option determines how results are grouped.

Available values are:

- `controller` — aggregate results by character and controller classification.
- `player` — aggregate results by individual player.

The default is `controller`.

For example:

```bash
python main.py --dir ./replays --aggregate controller
```

or:

```bash
python main.py --dir ./replays --aggregate player
```

Short form:

```bash
python main.py -d ./replays -a player
```

### Generate a report

Use `--report` or `-r` to generate the console report:

```bash
python main.py --dir ./replays --report
```

The report includes dataset-level information as well as aggregated gameplay statistics.

### Generate plots

Use `--plots` or `-p` to generate visualizations:

```bash
python main.py --dir ./replays --plots
```

Plots are generated for the available gameplay metrics and are organized by character and analysis type.

### Save scan results

Use `--save` or `-s` to save the parsed scan results to a cache file:

```bash
python main.py --dir ./replays --save
```

Saving results is useful when working with a large replay collection because subsequent analysis can be performed without repeating the expensive replay-scanning stage.

### Number of threads

The `--threads` or `-t` option controls the number of threads/workers used for processing.

For example:

```bash
python main.py --dir ./replays --threads 8
```

The default value is `max`, which uses the maximum available number of CPU threads/workers supported by the application.

To use a single worker:

```bash
python main.py --dir ./replays --threads 1
```

## Command-Line Reference

| Argument | Short | Description | Default |
|---|---|---|---|
| `--dir` | `-d` | Directory containing `.slp` or `.zip` files | Required* |
| `--file` | `-f` | Specific `.slp` or `.zip` file to process | Required* |
| `--aggregate` | `-a` | Aggregation method: `controller` or `player` | `controller` |
| `--report` | `-r` | Enable report generation | Disabled |
| `--plots` | `-p` | Enable plotting output | Disabled |
| `--save` | `-s` | Save parsed scan results to a cache file | Disabled |
| `--threads` | `-t` | Number of workers to use or `max` | `max` |

\* `--dir` and `--file` are mutually exclusive, and exactly one must be supplied.

## Example Workflows

### Scan and generate a report

```bash
python main.py \
    --dir ./replays \
    --report
```

### Scan, save, and generate plots

```bash
python main.py \
    --dir ./replays \
    --save \
    --plots
```

### Full analysis

```bash
python main.py \
    --dir ./replays \
    --aggregate controller \
    --report \
    --plots \
    --save \
    --threads max
```

### Analyze individual players

```bash
python main.py \
    --dir ./replays \
    --aggregate player \
    --report \
    --plots
```

### Process a single replay

```bash
python main.py \
    --file ./replays/game.slp \
    --report
```

## Analysis Methodology

### Controller Classification

Controller classification is based on the observed analog-stick coordinate distributions in replay data.

The classifier examines several properties of the input data, including:

- Number of unique coordinates
- Number of unique coordinates after extracting repeated holds
- Frequency of small off-axis inputs
- Number of unique stick-rim coordinates
- Angular entropy
- Proportion of inputs occurring near the edge of the analog range

These characteristics are used to distinguish conventional analog controllers from controllers that produce a more discrete input space.

The classification system intentionally includes a `Possibly Digital` category for datasets that exhibit characteristics associated with digital controllers without meeting the stronger classification criteria.

### Statistical Comparisons

When comparing two distributions, the project uses both hypothesis tests and effect-size measurements.

The Kolmogorov-Smirnov test evaluates whether the two distributions differ, while the Mann-Whitney U test evaluates whether observations from one distribution tend to be systematically larger or smaller than observations from the other.

Effect sizes provide additional information about the magnitude and nature of differences. The project calculates:

- **Cliff's delta** — measures stochastic dominance between two distributions.
- **Hodges-Lehmann estimator** — estimates the typical pairwise difference between observations from two distributions.
- **IQR difference** — compares the spread of the middle 50% of each distribution.
- **Skewness difference** — compares distribution asymmetry.

Bootstrap resampling is used to estimate uncertainty and construct confidence intervals for these statistics.

For large datasets, specialized implementations and sampling strategies are used to prevent computational costs from growing unnecessarily with the Cartesian product of the two datasets.

## Performance and Scalability

Replay analysis can involve processing very large numbers of frame-level observations. The project therefore incorporates several strategies to improve scalability.

### Parallel Processing

Independent replay files can be processed concurrently. The application exposes the number of workers through the `--threads` argument.

Plot generation is similarly organized as independent tasks and scheduled across workers. Plot tasks are weighted according to their expected computational cost so that larger tasks can be scheduled earlier.

### Memory and Computational Considerations

Several statistical calculations, particularly pairwise effect-size calculations, can require operations proportional to:

```text
len(A) × len(B)
```

The project therefore uses different algorithms depending on dataset size. Smaller datasets can use direct calculations, while larger datasets use more memory-efficient approaches or sampling.

Bootstrap iteration counts are also adjusted according to dataset size to balance statistical precision against computational cost.

## Output

Generated plots are organized into directories according to the analysis being performed.

For example:

```text
plots/
└── characters/
    ├── Fox/
    │   ├── nair_1_analog.png
    │   ├── nair_1_digital.png
    │   ├── nair_1_overlay.png
    │   └── ...
    └── Marth/
        └── ...
```

Reports are printed to the console and logging output is maintained separately for scanning and plotting operations.

Cached scan results can be used to preserve the results of an expensive replay-processing operation for later analysis.

## Engineering Considerations

The project was developed with several software engineering principles in mind:

- **Modular architecture** — replay parsing, metric extraction, statistical analysis, plotting, reporting, and file management are separated into independent modules.
- **Reusable data structures** — scan results and plotting operations are represented using structured task/data objects.
- **Parallel processing** — expensive independent operations can be distributed across multiple workers.
- **Caching** — parsed replay data can be persisted and reused.
- **Error handling** — individual processing and plotting failures are logged without necessarily terminating the entire workload.
- **Configurable analysis** — thresholds, statistical parameters, plotting parameters, and processing limits are centralized in configuration.
- **Scalable statistical methods** — algorithms change depending on dataset size to avoid excessive memory and computation requirements.
- **Reproducible analysis** — cached scan results allow reports and visualizations to be regenerated from the same underlying dataset.

## Project Goals

This project was developed as part of an analysis of controller types and gameplay behavior in competitive Super Smash Bros. Melee.

Beyond the gameplay analysis itself, the project serves as an application of **data science and data engineering techniques to a large-scale, frame-level dataset**. The pipeline combines data extraction, preprocessing, feature engineering, classification, aggregation, statistical testing, uncertainty estimation, visualization, and parallel computation.

The architecture is intended to make it possible to process large replay collections efficiently while keeping the individual stages of the analysis independently reusable.
