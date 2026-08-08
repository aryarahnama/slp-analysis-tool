from collections import Counter, defaultdict

import melee
import numpy as np

from state.player_state import PlayerState
from config import (STAGE_MAP, CHARACTER_MAP, MIN_ANIMATIONS,
                    EARLY_AERIAL_FRAMES, UNKNOWN_CHARACTER, SPACIES, SPACIE_METRICS)
from utils.misc import normalize_connect_code, process_analog_stick, dd_list, ddd_list, build_frame_hist
from features.metrics import is_handwarmer, classify_controller


def scan_slp_stream(console, actions_by_port, coords_by_port, coords_by_port_raw, stocks_by_port, codes_by_port, chars_by_port):
    """Process a Slippi game stream and collect per-player game state.

    Args:
        console: The console or game-state stream used to iterate through
            successive frames of the game.
        actions_by_port: Mapping of player ports to PlayerState objects used
            to store animations, inputs, positions, and derived metrics.
        coords_by_port: Mapping of player ports to lists storing processed
            analog stick coordinates for each frame.
        coords_by_port_raw: Mapping of player ports to lists storing raw
            analog stick coordinates for each frame.
        stocks_by_port: Mapping of player ports to lists storing the player's
            stock count for each frame.
        codes_by_port: Mapping of player ports to player connect codes.
        chars_by_port: Mapping of player ports to the player's character.

    Returns:
        A tuple containing:
            local_is_doubles: Whether the game uses teams.
            local_game_start_at: The frame at which the game starts.
            stage: The normalized stage identifier.
    """

    local_is_doubles = None
    local_game_start_at = None
    stage = None

    while (gamestate := console.step()) is not None:
        if local_is_doubles is None:
            local_is_doubles = gamestate.is_teams

        if local_game_start_at is None:
            local_game_start_at = gamestate.startAt

        if stage is None:
            stage = STAGE_MAP.get(
                gamestate.stage.value if gamestate.stage else 0)

        for port, player in gamestate.players.items():
            if player is None:
                continue

            # metadata
            if port not in codes_by_port:
                code = getattr(player, "connectCode", None)
                if code is not None:
                    code = normalize_connect_code(code)

                codes_by_port[port] = code
                chars_by_port[port] = CHARACTER_MAP.get(
                    player.character.value if player.character else UNKNOWN_CHARACTER)

            # collect stick coords
            x_raw, y_raw = player.controller_state.raw_main_stick
            x_deadzone, y_deadzone = process_analog_stick(x_raw, y_raw, True)
            x, y = process_analog_stick(x_raw, y_raw, False)

            coords_by_port_raw[port].append((int(x_raw), int(y_raw)))
            coords_by_port[port].append((float(x), float(y)))
            stocks_by_port[port].append(player.stock)

            # ACTION TRACKING
            state = actions_by_port[port]

            current_anim = player.action.value
            current_frame_counter = getattr(player, "action_frame", 0)

            controller_state = player.controller_state

            state.l_digital.append(
                controller_state.button[melee.Button.BUTTON_L])
            state.r_digital.append(
                controller_state.button[melee.Button.BUTTON_R])

            state.animations.append(current_anim)
            state.frame_counters.append(current_frame_counter)
            state.processed_coords.append((float(x_deadzone), float(y_deadzone)))
            state.hitlag_left.append(player.hitlag_left)
            state.hitstun_frames_left.append(player.hitstun_frames_left)
            state.on_ground.append(player.on_ground)

            x_position = float(player.position.x) if player.position is not None else 0.0
            y_position = float(player.position.y) if player.position is not None else 0.0

            state.positions_x.append(x_position)
            state.positions_y.append(y_position)

            if len(state.animations) <= MIN_ANIMATIONS:
                continue

            previous_anim = state.animations[-2]
            previous_frame_counter = state.frame_counters[-2]

            if not state.is_new_action(previous_anim, previous_frame_counter, current_anim, current_frame_counter):
                continue

            state.track_dash_dances()
            state.track_wavedashes()
            state.track_sdi_windows()
            state.track_early_aerials()
            state.track_tilts()

            if chars_by_port[port] in SPACIES:
                state.track_jc_shines()

    return local_is_doubles, local_game_start_at, stage


def analyze_slp_results(file_path, actions_by_port, coords_by_port, codes_by_port, chars_by_port, local_game_start_at, stage):
    """Aggregate player state into per-game metrics and analysis results.

    Args:
        file_path: Path to the SLP file being analyzed.
        actions_by_port: Mapping of player ports to PlayerState objects
            containing tracked actions and derived metrics.
        coords_by_port: Mapping of player ports to processed analog stick
            coordinates used to classify the controller.
        codes_by_port: Mapping of player ports to normalized player connect
            codes.
        chars_by_port: Mapping of player ports to normalized character
            identifiers.
        local_game_start_at: Frame at which the game starts.
        stage: Normalized stage identifier for the game.

    Returns:
        A tuple containing:
            local_rows: Per-player rows of aggregated game metrics.
            local_stage_counts: Counts of games played on each stage.
            local_char_counts: Counts of players using each character.
            local_player_stats: Per-player counts of controller
                classifications.
            local_plot_data: Metric data grouped by player and controller
                classification for plotting and further analysis.
    """

    local_rows = []
    local_stage_counts = Counter()
    local_char_counts = Counter()
    local_player_stats = defaultdict(Counter)

    local_plot_data = {
        "controller": defaultdict(ddd_list),
        "player": defaultdict(dd_list),
    }

    local_stage_counts[stage] += 1

    # finalize per-player state
    for state in actions_by_port.values():
        state.finalize()

    for port in coords_by_port:
        coords = coords_by_port[port]
        result = classify_controller(coords)

        code = codes_by_port[port]
        char = chars_by_port[port]

        local_char_counts[char] += 1
        local_player_stats[code][result] += 1

        wavedash_count = actions_by_port[port].wavedash_count
        waveland_count = actions_by_port[port].waveland_count
        dash_dances = actions_by_port[port].dash_dances
        dash_dance_lengths = actions_by_port[port].dash_dance_lengths
        dash_dance_distances = actions_by_port[port].dash_dance_distances
        intangible_ledgedashes = actions_by_port[port].intangible_ledgedashes
        ledgedash_sd_proportion = actions_by_port[port].ledgedash_sd_proportion
        no_impact_lands = actions_by_port[port].no_impact_lands
        total_galint = np.sum(actions_by_port[port].galint)
        mean_galint = actions_by_port[port].mean_galint
        max_galint = actions_by_port[port].max_galint
        sdi_moves = actions_by_port[port].sdi_moves
        sdi_opportunities = actions_by_port[port].sdi_opportunities
        sdi_proportion = actions_by_port[port].sdi_proportion
        sdi_inputs = actions_by_port[port].sdi_inputs
        sdi_magnitudes = actions_by_port[port].sdi_magnitudes
        jc_shines = actions_by_port[port].jc_shine_count
        jc_shine_frames = actions_by_port[port].jc_shine_frames
        crouch_uptilts = actions_by_port[port].crouch_uptilt_count
        crouch_uptilt_frames = actions_by_port[port].crouch_uptilt_frames
        pivot_uptilts = actions_by_port[port].pivot_uptilt_count
        pivot_uptilt_frames = actions_by_port[port].pivot_uptilt_frames
        pivot_forwardtilts = actions_by_port[port].pivot_forwardtilt_count
        pivot_forwardtilt_frames = actions_by_port[port].pivot_forwardtilt_frames
        pivot_downtilts = actions_by_port[port].pivot_downtilt_count
        pivot_downtilt_frames = actions_by_port[port].pivot_downtilt_frames

        wavedash_frames = actions_by_port[port].wavedash_frames
        wavedash_angles = actions_by_port[port].wavedash_angles
        ledgedash_angles = actions_by_port[port].ledgedash_angles
        galint = actions_by_port[port].galint

        num_dash_dances_per_game = [dash_dances]
        num_wavedashes_per_game = [wavedash_count]
        num_intangible_ledgedashes_per_game = [intangible_ledgedashes]
        num_crouch_uptilts_per_game = [crouch_uptilts]
        num_pivot_uptilts_per_game = [pivot_uptilts]
        num_pivot_forwardtilts_per_game = [pivot_forwardtilts]
        num_pivot_downtilts_per_game = [pivot_downtilts]
        num_jc_shines_per_game = [jc_shines]
        ledgedash_sd_proportion = [ledgedash_sd_proportion]
        sdi_proportion = [sdi_proportion]
        valid_wd_angles = [a for a in wavedash_angles if a is not None]
        valid_ld_angles = [a for a in ledgedash_angles if a is not None]
        avg_wd_frame = np.mean(wavedash_frames) if len(wavedash_frames) > 0 else 0.0
        avg_wd_angle = np.mean(valid_wd_angles) if len(valid_wd_angles) > 0 else 0.0
        avg_ld_angle = np.mean(valid_ld_angles) if len(valid_ld_angles) > 0 else 0.0

        nair_data = actions_by_port[port].nair_frames
        fair_data = actions_by_port[port].fair_frames
        bair_data = actions_by_port[port].bair_frames
        uair_data = actions_by_port[port].uair_frames
        dair_data = actions_by_port[port].dair_frames

        nair_hist = build_frame_hist(nair_data)
        fair_hist = build_frame_hist(fair_data)
        bair_hist = build_frame_hist(bair_data)
        uair_hist = build_frame_hist(uair_data)
        dair_hist = build_frame_hist(dair_data)

        metrics = {
            "num_dash_dances_per_game": num_dash_dances_per_game,
            "dash_dance_lengths": dash_dance_lengths,
            "dash_dance_distances": dash_dance_distances,
            "num_wavedashes_per_game": num_wavedashes_per_game,
            "wavedash_frames": wavedash_frames,
            "wavedash_angles": valid_wd_angles,
            "num_intangible_ledgedashes_per_game": num_intangible_ledgedashes_per_game,
            "ledgedash_sd_proportion": ledgedash_sd_proportion,
            "ledgedash_angles": valid_ld_angles,
            "galint": galint,
            "sdi_proportion": sdi_proportion,
            "sdi_inputs": sdi_inputs,
            "sdi_magnitudes": sdi_magnitudes,
            "num_crouch_uptilts_per_game": num_crouch_uptilts_per_game,
            "crouch_uptilt_frames": crouch_uptilt_frames,
            "num_pivot_uptilts_per_game": num_pivot_uptilts_per_game,
            "pivot_uptilt_frames": pivot_uptilt_frames,
            "num_pivot_forwardtilts_per_game": num_pivot_forwardtilts_per_game,
            "pivot_forwardtilt_frames": pivot_forwardtilt_frames,
            "num_pivot_downtilts_per_game": num_pivot_downtilts_per_game,
            "pivot_downtilt_frames": pivot_downtilt_frames,
            "num_jc_shines_per_game": num_jc_shines_per_game,
            "jc_shine_frames": jc_shine_frames,
            "nair_frames": nair_data,
            "fair_frames": fair_data,
            "bair_frames": bair_data,
            "uair_frames": uair_data,
            "dair_frames": dair_data,
        }

        if code is not None:
            for key, values in metrics.items():
                if key in SPACIE_METRICS and char not in SPACIES:
                    continue

                local_plot_data["player"][code][key].extend(values)

        if result is not None:
            for key, values in metrics.items():
                if key in SPACIE_METRICS and char not in SPACIES:
                    continue

                local_plot_data["controller"][char][result][key].extend(values)

        local_row = {
            "file": file_path,
            "start_at": local_game_start_at,
            "stage": stage,
            "character": char,
            "connect_code": code,
            "classification": result,
            "dash_dances": dash_dances,
            "wavedashes": wavedash_count,
            "wavelands": waveland_count,
            "avg_wavedash_frame": avg_wd_frame,
            "avg_wavedash_angle": avg_wd_angle,
            "intangible_ledgedashes": intangible_ledgedashes,
            "avg_ledgedash_angle": avg_ld_angle,
            "no_impact_lands": no_impact_lands,
            "total_galint": total_galint,
            "mean_galint": mean_galint,
            "max_galint": max_galint,
            "sdi_moves": sdi_moves,
            "sdi_opportunities": sdi_opportunities,
            "dash_dance_lengths_sum": sum(dash_dance_lengths),
            "dash_dance_lengths_count": len(dash_dance_lengths),
            "dash_dance_distances_sum": sum(dash_dance_distances),
            "dash_dance_distances_count": len(dash_dance_distances),
            "wavedash_frame_sum": sum(wavedash_frames),
            "wavedash_frame_count": len(wavedash_frames),
            "wavedash_angle_sum": sum(valid_wd_angles),
            "wavedash_angle_count": len(valid_wd_angles),
            "ledgedash_angle_sum": sum(valid_ld_angles),
            "ledgedash_angle_count": len(valid_ld_angles),
            "sdi_inputs_sum": sum(sdi_inputs),
            "sdi_inputs_count": len(sdi_inputs),
            "sdi_magnitudes_sum": sum(sdi_magnitudes),
            "sdi_magnitudes_count": len(sdi_magnitudes)
        }

        for i in range(1, EARLY_AERIAL_FRAMES + 1):
            local_row[f"nair_{i}"] = nair_hist[i]
            local_row[f"fair_{i}"] = fair_hist[i]
            local_row[f"bair_{i}"] = bair_hist[i]
            local_row[f"uair_{i}"] = uair_hist[i]
            local_row[f"dair_{i}"] = dair_hist[i]

        local_rows.append(local_row)

    return local_rows, local_stage_counts, local_char_counts, local_player_stats, local_plot_data


def process_slp(file_path):
    """Process and analyze a single Slippi replay file.

    Args:
        file_path: Path to the SLP file to process.

    Returns:
        False if the game is a doubles match or a handwarmer and should
        therefore be excluded from analysis. Otherwise, returns the
        aggregated analysis results produced by analyze_slp_results.
    """

    console = melee.Console(is_dolphin=False, path=file_path)
    console.connect()

    # per-player storage
    coords_by_port = defaultdict(list)
    coords_by_port_raw = defaultdict(list)
    stocks_by_port = defaultdict(list)
    actions_by_port = defaultdict(PlayerState)
    codes_by_port = {}
    chars_by_port = {}

    local_is_doubles, local_game_start_at, stage = scan_slp_stream(console, actions_by_port, coords_by_port, coords_by_port_raw, stocks_by_port, codes_by_port, chars_by_port)

    if local_is_doubles or is_handwarmer(coords_by_port, stocks_by_port):  # filter
        return False

    return analyze_slp_results(file_path, actions_by_port, coords_by_port, codes_by_port, chars_by_port, local_game_start_at, stage)
