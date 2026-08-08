from typing import List, Tuple

import math

from config import (DZ_THRESHOLD_COORD, FRAMES_PER_SECOND, SECONDS_PER_MINUTE, MAX_DIGITAL_COORDS, MAX_UNIQUE_DIGITAL_COORDS, MAX_POSSIBLY_DIGITAL_COORDS,
                    MAX_UNIQUE_POSSIBLY_DIGITAL_COORDS, MAX_POSSIBLY_DIGITAL_ENTROPY, SMALL_OFF_AXIS_THRESHOLD, EPSILON, MIN_ANALOG_UNIQUE, MIN_ANALOG_RATIO, RIM_COORD_MAX, MAX_RIM_PROP, THREE_MINUTES)


Coord = Tuple[float, float]


def is_in_deadzone(x, y):
    """Determine whether an analog stick position is within the deadzone.

    Args:
        x: Horizontal analog stick coordinate.
        y: Vertical analog stick coordinate.

    Returns:
        True if both coordinates are within the configured deadzone
        threshold, otherwise False.
    """

    return abs(x) < DZ_THRESHOLD_COORD and abs(y) < DZ_THRESHOLD_COORD


def is_handwarmer(coords_by_port, stocks_by_port, min_seconds=SECONDS_PER_MINUTE):
    """Determine whether a replay should be classified as a handwarmer.

    A replay is classified as a handwarmer if it is empty, shorter than the
    minimum duration, or contains a player who remains in the analog stick
    deadzone for an extended period while still alive.

    Args:
        coords_by_port: Mapping of player ports to processed analog stick
            coordinates for each frame.
        stocks_by_port: Mapping of player ports to stock counts for each
            frame.
        min_seconds: Minimum replay duration in seconds. Replays shorter than
            this duration are classified as handwarmers.

    Returns:
        True if the replay meets the criteria for a handwarmer, otherwise
        False.
    """

    if not coords_by_port:
        return True

    total_frames = len(next(iter(coords_by_port.values())))

    # short game check
    if total_frames < (FRAMES_PER_SECOND * min_seconds):
        return True

    # per-player deadzone streak check
    for port in coords_by_port:
        deadzone_frames = 0

        coords = coords_by_port[port]
        stocks = stocks_by_port[port]

        for i in range(total_frames):
            # reset if dead
            if stocks[i] == 0:
                deadzone_frames = 0
                continue

            x, y = coords[i]

            if is_in_deadzone(x, y):
                deadzone_frames += 1
            else:
                deadzone_frames = 0

            if deadzone_frames > (10 * FRAMES_PER_SECOND):  # 10 seconds
                return True

    return False


def extract_holds(coords: List[Coord], min_length: int = 2) -> List[Coord]:
    """Extract analog stick coordinates that are held for multiple frames.

    Identifies consecutive runs of identical coordinates and records the
    coordinate for each run whose length meets the minimum threshold.

    Args:
        coords: Sequence of analog stick coordinates.
        min_length: Minimum number of consecutive frames required for a
            coordinate to be considered a hold.

    Returns:
        A list containing the starting coordinate of each qualifying hold.
    """

    if not coords:
        return []
    elif min_length <= 1:
        return coords.copy()

    holds = []
    i = 0

    while i < len(coords):
        start = i

        # walk while same coord repeats
        while i + 1 < len(coords) and coords[i] == coords[i + 1]:
            i += 1

        length = i - start + 1

        if length >= min_length:
            holds.append(coords[start])

        i += 1

    return holds


def get_angles(coords):
    """Calculate the polar angle of each non-neutral stick position.

    Args:
        coords: Sequence of analog stick coordinates.

    Returns:
        A list of angles in radians calculated from the positive horizontal
        axis. Neutral stick positions are excluded.
    """

    angles = []

    for x, y in coords:
        if x == 0 and y == 0:
            continue

        angle = math.atan2(y, x)
        angles.append(angle)

    return angles


def angle_entropy(angles, bins=16):
    """Calculate the entropy of a distribution of stick angles.

    Angles are divided into equally sized bins around the unit circle, and
    the resulting probability distribution is used to calculate Shannon
    entropy.

    Args:
        angles: Sequence of stick angles in radians.
        bins: Number of angular bins used to construct the distribution.

    Returns:
        The Shannon entropy of the angular distribution, or 0 if no angles
        are provided.
    """

    if not angles:
        return 0

    hist = [0] * bins
    for a in angles:
        idx = int((a + math.pi) / (2 * math.pi) * bins)
        idx = min(idx, bins - 1)
        hist[idx] += 1

    total = sum(hist)
    probs = [h / total for h in hist if h > 0]

    return -sum(p * math.log(p) for p in probs)


def count_small_off_axis_coords(coords):
    """Count stick coordinates with small off-axis components.

    Identifies coordinates where the smaller absolute component is nonzero
    but remains below the configured off-axis threshold.

    Args:
        coords: Sequence of analog stick coordinates.

    Returns:
        A tuple containing:
            count: Total number of coordinates meeting the off-axis
                criterion.
            unique_count: Number of unique coordinates meeting the criterion.
    """

    count = 0
    unique = set()

    for x, y in coords:
        min_abs = min(abs(x), abs(y))

        if 0 < min_abs < SMALL_OFF_AXIS_THRESHOLD:
            count += 1
            unique.add((x, y))

    return count, len(unique)


def count_rim_coords(coords):
    """Count unique analog stick coordinates located on the stick rim.

    A coordinate is considered a rim coordinate when its distance from the
    origin, calculated using a small epsilon adjustment, is at least one.

    Args:
        coords: Sequence of analog stick coordinates.

    Returns:
        The number of unique coordinates located on or beyond the stick rim.
    """

    rim_coords = set()

    for x, y in coords:
        dist = math.sqrt((abs(x) + EPSILON)**2 + (abs(y) + EPSILON)**2)

        if dist >= 1.0:
            rim_coords.add((x, y))

    return len(rim_coords)


def is_digital_controller(coords):
    """Determine whether analog stick data is consistent with a digital controller.

    Uses the proportion of unique rim coordinates and the presence of small
    off-axis coordinates to distinguish digital controller input from
    continuous analog input. Shorter samples receive an additional rim
    proportion adjustment to account for reduced observation time.

    Args:
        coords: Sequence of processed analog stick coordinates.

    Returns:
        True if the stick input meets the configured digital-controller
        criteria, otherwise False.
    """

    if not coords:
        return False

    small_count, small_unique = count_small_off_axis_coords(coords)
    ratio = small_count / len(coords)

    if ratio >= MIN_ANALOG_RATIO and small_unique >= MIN_ANALOG_UNIQUE:
        return False

    rim_count = count_rim_coords(coords)
    rim_prop = rim_count / RIM_COORD_MAX

    # boost for short games
    if len(coords) < THREE_MINUTES:
        boost = 1 + (THREE_MINUTES - len(coords)) / THREE_MINUTES
        rim_prop *= boost

    return rim_prop < MAX_RIM_PROP


def classify_controller(coords) -> str:
    """Classify a controller based on its analog stick input.

    Uses the number of unique coordinates, repeated-coordinate holds, angular
    entropy, and digital-controller heuristics to classify the input as
    Digital, Possibly Digital, or Analog.

    Args:
        coords: Sequence of processed analog stick coordinates collected from
            a player during a replay.

    Returns:
        A controller classification: "Digital", "Possibly Digital", or
        "Analog".
    """

    unique = len(set(coords))

    holds = extract_holds(coords, min_length=2)
    hold_unique = len(set(holds))

    angles = get_angles(coords)
    entropy = angle_entropy(angles)

    if (unique <= MAX_DIGITAL_COORDS and hold_unique <= MAX_UNIQUE_DIGITAL_COORDS) or is_digital_controller(coords):
        return "Digital"
    elif unique <= MAX_POSSIBLY_DIGITAL_COORDS and hold_unique <= MAX_UNIQUE_POSSIBLY_DIGITAL_COORDS and entropy < MAX_POSSIBLY_DIGITAL_ENTROPY:
        return "Possibly Digital"
    else:
        return "Analog"
