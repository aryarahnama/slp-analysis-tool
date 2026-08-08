from dataclasses import dataclass, field
from typing import List, Tuple

import math
import melee
import numpy as np

from config import (DASH_DANCE_FRAMES, WAVEDASH_FRAMES, EARLY_AERIAL_FRAMES, CROUCH_UPTILT_FRAMES, PIVOT_FRAMES, PIVOT_LOOKBACK, JC_SHINE_FRAMES, LEDGEDASH_FRAMES, FRAMES_PER_SECOND, DZ_THRESHOLD_COORD)


Coord = Tuple[float, float]


@dataclass(slots=True)
class PlayerState:
    animations: List[int] = field(default_factory=list)
    frame_counters: List[int] = field(default_factory=list)
    processed_coords: List[Coord] = field(default_factory=list)
    hitlag_left: List[int] = field(default_factory=list)
    hitstun_frames_left: List[int] = field(default_factory=list)
    on_ground: List[bool] = field(default_factory=list)
    positions_x: List[float] = field(default_factory=list)
    positions_y: List[float] = field(default_factory=list)
    l_digital: List[bool] = field(default_factory=list)
    r_digital: List[bool] = field(default_factory=list)

    dash_dances: int = 0
    dash_dance_lengths: List[int] = field(default_factory=list)
    dash_dance_distances: List[float] = field(default_factory=list)

    wavedash_count: int = 0
    wavedash_frames: List[int] = field(default_factory=list)
    wavedash_angles: List[float] = field(default_factory=list)
    waveland_count: int = 0

    intangible_ledgedashes: int = 0
    ledgedash_angles: List[float] = field(default_factory=list)
    ledgedash_sd_proportion: float = 0.0
    no_impact_lands: int = 0
    galint: List[int] = field(default_factory=list)
    max_galint: int = 0
    mean_galint: float = 0.0

    sdi_moves: int = 0
    sdi_opportunities: int = 0
    sdi_proportion: float = 0.0
    sdi_inputs: List[int] = field(default_factory=list)
    sdi_magnitudes: List[float] = field(default_factory=list)

    nair_frames: List[int] = field(default_factory=list)
    fair_frames: List[int] = field(default_factory=list)
    bair_frames: List[int] = field(default_factory=list)
    uair_frames: List[int] = field(default_factory=list)
    dair_frames: List[int] = field(default_factory=list)

    crouch_uptilt_count: int = 0
    crouch_uptilt_frames: List[int] = field(default_factory=list)
    pivot_uptilt_count: int = 0
    pivot_uptilt_frames: List[int] = field(default_factory=list)

    pivot_forwardtilt_count: int = 0
    pivot_forwardtilt_frames: List[int] = field(default_factory=list)

    pivot_downtilt_count: int = 0
    pivot_downtilt_frames: List[int] = field(default_factory=list)

    jc_shine_count: int = 0
    jc_shine_frames: List[int] = field(default_factory=list)

    def __did_release_ledge(self, previous, current):
        """Determine whether the player has released the ledge.

        Args:
            previous: The player's action on the previous frame.
            current: The player's action on the current frame.

        Returns:
            True if the player transitions from an edge-catching or
            edge-hanging action to a falling action.
        """

        return (previous == melee.Action.EDGE_CATCHING.value or previous == melee.Action.EDGE_HANGING.value) and (melee.Action.FALLING.value <= current <= melee.Action.FALLING_BACKWARD.value)

    def __did_cliff_catch_end(self, previous, current):
        """Determine whether the player's cliff-catch state has ended.

        Args:
            previous: The player's action on the previous frame.
            current: The player's action on the current frame.

        Returns:
            True if the previous action was edge catching and the current
            action is no longer edge catching.
        """

        return (previous == melee.Action.EDGE_CATCHING.value) and (current != melee.Action.EDGE_CATCHING.value)

    def __is_landing(self, current):
        """Determine whether the player is in a landing state.

        Args:
            current: The player's action on the current frame.

        Returns:
            True if the current action is a normal or special landing.
        """

        return (current == melee.Action.LANDING.value) or (current == melee.Action.LANDING_SPECIAL.value)
    
    def __is_crouch(self, current):
        """Determine whether the player is in a crouching state.

        Args:
            current: The player's action on the current frame.

        Returns:
            True if the current action falls within the crouching action range.
        """

        return (melee.Action.CROUCH_START.value <= current <= melee.Action.CROUCH_END.value)
    
    def __is_shine(self, current):
        """Determine whether the player is performing a shine.

        Args:
            current: The player's action on the current frame.

        Returns:
            True if the current action falls within the shine action range.
        """

        return (melee.Action.DOWN_B_GROUND_START.value <= current <= melee.Action.DOWN_B_AIR.value)

    def __is_airborne(self, on_ground, current):
        """Determine whether the player is airborne.

        Args:
            on_ground: Whether the player is considered to be on the ground.
            current: The player's action on the current frame.

        Returns:
            True if the player is not grounded or is in an aerial action.
        """

        return (not on_ground) or (melee.Action.KNEE_BEND.value < current <= melee.Action.FALLING_AERIAL_BACKWARD.value)

    def __did_no_impact_land(self, previous, current):
        """Determine whether the player performed a no-impact land.

        Args:
            previous: The player's action on the previous frame.
            current: The player's action on the current frame.

        Returns:
            True if the player transitions from an aerial or special falling action
            to a standing action.
        """

        # JUMPING_ARIAL_BACKWARD could be upper bound
        return (melee.Action.JUMPING_FORWARD.value <= previous <= melee.Action.SPECIAL_FALL_BACK.value) and (current == melee.Action.STANDING.value)

    def __is_dash_dance_frame(self, last3):
        """Determine whether three consecutive frames form a dash dance.

        Args:
            last3: The player's actions on the three most recent frames.

        Returns:
            True if the three frames contain a dash-turn-dash sequence or
            three consecutive dashing actions (ucf dashback).

        Credit: Fizzi/Slippi
        """

        a, b, c = last3
        return (a == melee.Action.DASHING.value and b == melee.Action.TURNING.value and c == melee.Action.DASHING.value) or (a == b == c == melee.Action.DASHING.value)
    
    def __is_pivot_frame(self, last3):
        """Determine whether three consecutive frames contain a pivot.

        Args:
            last3: The player's actions on the three most recent frames.

        Returns:
            True if the frames contain a dash-to-turn transition followed
            by an action other than dashing.

        Credit: cbartsch/slippic
        """

        a, b, c = last3
        return (a == melee.Action.DASHING.value and b == melee.Action.TURNING.value and c != melee.Action.DASHING.value)
    
    def __is_airdodge_sd(self, recent_frames):
        """Determine whether recent frames contain an air dodge self-destruct.

        A landing within the provided frames prevents the sequence from
        being classified as an air dodge self-destruct.

        Args:
            recent_frames: A sequence of the player's recent actions.

        Returns:
            True if the frames contain an air dodge followed by a dead-fall
            sequence without a landing.
        """

        has_landing = any(self.__is_landing(anim) for anim in recent_frames)

        if has_landing:
            return False
        
        targets = [
            [melee.Action.AIRDODGE.value, melee.Action.DEAD_FALL.value],
        ]

        found_target = any(
            recent_frames[i:i+len(target)] == target
            for target in targets
            for i in range(len(recent_frames) - len(target) + 1)
        )

        return found_target

    def __count_airborne_frames(self, recent_frames, on_ground):
        """Count the number of airborne frames in a sequence.

        Args:
            recent_frames: A sequence of the player's recent actions.
            on_ground: Grounded-state values corresponding to each action.

        Returns:
            The number of frames during which the player was airborne.
        """

        return sum(self.__is_airborne(grounded, anim) for anim, grounded in zip(recent_frames, on_ground))
    
    def __count_frames_since_pivot(self, recent_frames):
        """Determine the number of frames since the most recent pivot.

        Args:
            recent_frames: A sequence of the player's recent actions.

        Returns:
            The number of frames since the detected pivot, or None if no
            pivot is found in the provided frames.
        """

        frames_since_pivot = None

        for i in range(len(recent_frames) - PIVOT_FRAMES, -1, -1):
            window = recent_frames[i:i + PIVOT_FRAMES]

            if self.__is_pivot_frame(window):
                pivot_idx = i + PIVOT_FRAMES // 2

                frames_since_pivot = (len(recent_frames) - 1) - pivot_idx
                break

        return frames_since_pivot
    
    def __count_frames_since(self, num_frames_to_check, animation):
        """Determine the number of frames since the most recent action.

        Args:
            num_frames_to_check: The maximum number of preceding frames to
                search.
            animation: The animation to search for.

        Returns:
            The number of frames since the specified animation began, or
            None if the animation is not found within the search range.
        """

        frames_since_jump = None

        for k in range(1, num_frames_to_check + 1):
            idx = len(self.animations) - 1 - k
            if idx < 0:
                break

            anim, frame_counter = self.animations[idx], self.frame_counters[idx]
            next_anim, next_frame_counter = self.animations[idx + 1], self.frame_counters[idx + 1]

            if anim == animation and self.is_new_action(anim, frame_counter, next_anim, next_frame_counter):
                frames_since_jump = k
                break

        return frames_since_jump

    def __is_wavedash_initiation_animation(self, animation):
        """Determine whether an animation can initiate a wavedash.

        Args:
            animation: The player's current animation.

        Returns:
            True if the animation is an air dodge or an animation within
            the aerial action range that can initiate a wavedash.

        Credit: Fizzi/Slippi
        """

        return (animation == melee.Action.AIRDODGE.value) or (melee.Action.KNEE_BEND.value <= animation <= melee.Action.FALLING_AERIAL_BACKWARD.value)
        
    def __is_new_digital_press(self, previous_l, previous_r, current_l, current_r):
        """Determine whether a new digital L or R press occurred.

        Args:
            previous_l: L button state on the previous frame.
            previous_r: R button state on the previous frame.
            current_l: L button state on the current frame.
            current_r: R button state on the current frame.

        Returns:
            True if L or R is pressed on the current frame but was not
            pressed on the previous frame.
        """

        return (current_l and not previous_l) or (current_r and not previous_r)

    def __get_sdi_region(self, x, y):
        """Classify an SDI input into a directional region.

        Args:
            x: The horizontal component of the SDI input.
            y: The vertical component of the SDI input.

        Returns:
            A region label representing the input direction: ``DZ`` for
            dead zone, ``N``, ``S``, ``E``, or ``W`` for cardinal directions,
            ``NE``, ``NW``, ``SE``, or ``SW`` for diagonals, or ``TILT`` for
            inputs that do not meet the thresholds for another region.

        Credit: altf4/enforcer 
        """

        dz = DZ_THRESHOLD_COORD

        if abs(x) <= dz and abs(y) <= dz:
            return "DZ"

        mag = math.sqrt(x*x + y*y)

        if x >= dz and y >= dz and mag >= 0.7:
            return "NE"
        if x >= dz and y <= -dz and mag >= 0.7:
            return "SE"
        if x <= -dz and y <= -dz and mag >= 0.7:
            return "SW"
        if x <= -dz and y >= dz and mag >= 0.7:
            return "NW"

        if y >= 0.7:
            return "N"
        if x >= 0.7:
            return "E"
        if y <= -0.7:
            return "S"
        if x <= -0.7:
            return "W"

        return "TILT"

    def __is_cardinal(self, region):
        """Determine whether an SDI region is cardinal.

        Args:
            region: The SDI region label.

        Returns:
            True if the region represents a cardinal direction.

        Credit: altf4/enforcer
        """

        return region in ("N", "S", "E", "W")

    def __is_diagonal(self, region):
        """Determine whether an SDI region is diagonal.

        Args:
            region: The SDI region label.

        Returns:
            True if the region represents a diagonal direction.

        Credit: altf4/enforcer
        """

        return region in ("NE", "NW", "SE", "SW")

    def __is_region_adjacent(self, region_a, region_b):
        """Determine whether a cardinal and diagonal region are adjacent.

        Args:
            region_a: The first SDI region label.
            region_b: The second SDI region label.

        Returns:
            True if the two regions are adjacent cardinal and diagonal
            directions.

        Credit: altf4/enforcer
        """

        cardinal_to_diagonal = {
            "N":  {"NE", "NW"},
            "E":  {"NE", "SE"},
            "S":  {"SE", "SW"},
            "W":  {"NW", "SW"},
        }

        diagonal_to_cardinal = {
            "NE": {"N", "E"},
            "NW": {"N", "W"},
            "SE": {"S", "E"},
            "SW": {"S", "W"},
        }

        if region_a in cardinal_to_diagonal:
            return region_b in cardinal_to_diagonal[region_a]

        if region_a in diagonal_to_cardinal:
            return region_b in diagonal_to_cardinal[region_a]

        return False

    def __is_diagonal_adjacent(self, region_a, region_b):
        """Determine whether two diagonal regions are adjacent.

        Args:
            region_a: The first diagonal SDI region.
            region_b: The second diagonal SDI region.

        Returns:
            True if the two diagonal regions are adjacent.

        Credit: altf4/enforcer
        """

        diagonal_adjacency = {
            "NE": {"NW", "SE"},
            "NW": {"NE", "SW"},
            "SE": {"NE", "SW"},
            "SW": {"NW", "SE"},
        }

        if region_a not in diagonal_adjacency:
            return False

        return region_b in diagonal_adjacency[region_a]

    def __is_valid_sdi_transition(self, prev_region, curr_region):
        """Determine whether an SDI region transition is valid.

        Valid transitions include a neutral-to-cardinal transition, a
        cardinal-to-adjacent-diagonal transition, and a transition between
        adjacent diagonal regions.

        Args:
            prev_region: The SDI region of the previous input.
            curr_region: The SDI region of the current input.

        Returns:
            True if the transition satisfies the valid SDI transition rules.

        Credit: altf4/enforcer
        """

        # Rule 1: neutral -> cardinal
        if prev_region == "DZ" and self.__is_cardinal(curr_region):
            return True

        # Rule 2: cardinal -> adjacent diagonal
        if self.__is_cardinal(prev_region) and self.__is_diagonal(curr_region):
            return self.__is_region_adjacent(prev_region, curr_region)

        # Rule 3: diagonal -> adjacent diagonal
        if self.__is_diagonal(prev_region) and self.__is_diagonal(curr_region):
            return self.__is_diagonal_adjacent(prev_region, curr_region)

        return False

    def __compute_angle_from_recent(self, recent_frames, recent_frame_counters, recent_coords, recent_ground, recent_l_digital, recent_r_digital):
        """Compute the directional angle from recent player state.

        The function searches recent frames for an appropriate reference
        point, prioritizing the first frame of the most recent air dodge,
        followed by the first frame of the most recent landing and the last
        airborne frame before a landing. If no suitable reference point is
        found, the most recent coordinate is used.

        Args:
            recent_frames: The player's recent animation states.
            recent_frame_counters: The animation frame counters corresponding
                to each recent frame.
            recent_coords: The player's recent controller-stick coordinates.
            recent_ground: The player's grounded state for each recent frame.
            recent_l_digital: The player's digital L-button state for each
                recent frame.
            recent_r_digital: The player's digital R-button state for each
                recent frame.

        Returns:
            The absolute directional angle in degrees, ranging from 0 to 90,
            or None if a non-zero directional vector cannot be determined.
        """

        airdodge_idx = landing_idx = airborne_idx = None
        idx = None
        x = y = None

        # 1. FIRST FRAME OF LAST AIRDODGE (best)
        for i in range(len(recent_frames) - 1, -1, -1):
            if i > 0:
                prev, prev_frame_counter, previous_l, previous_r = recent_frames[i - 1], recent_frame_counters[i - 1], recent_l_digital[i - 1], recent_r_digital[i - 1]
                anim, frame_counter, current_l, current_r = recent_frames[i], recent_frame_counters[i], recent_l_digital[i], recent_r_digital[i]

                if (self.is_new_action(prev, prev_frame_counter, anim, frame_counter) or self.__is_new_digital_press(previous_l, previous_r, current_l, current_r)) and anim == melee.Action.AIRDODGE.value:
                    airdodge_idx = i
                    x, y = recent_coords[i]
                    break
            elif i == 0 and recent_frames[i] == melee.Action.AIRDODGE.value:
                airdodge_idx = i
                x, y = recent_coords[i]
                break

        # 2. FIRST FRAME OF LAST LANDING
        if x is None or y is None:
            for i in range(len(recent_frames) - 1, -1, -1):
                if i > 0:
                    prev, prev_frame_counter, previous_l, previous_r = recent_frames[i - 1], recent_frame_counters[i - 1], recent_l_digital[i - 1], recent_r_digital[i - 1]
                    anim, frame_counter, current_l, current_r = recent_frames[i], recent_frame_counters[i], recent_l_digital[i], recent_r_digital[i]

                    if (self.is_new_action(prev, prev_frame_counter, anim, frame_counter) or self.__is_new_digital_press(previous_l, previous_r, current_l, current_r)) and self.__is_landing(anim):
                        landing_idx = i
                        x, y = recent_coords[i]
                        break
                elif i == 0 and self.__is_landing(recent_frames[i]):
                    landing_idx = i
                    x, y = recent_coords[i]
                    break

        # 3. LAST AIRBORNE FRAME BEFORE LANDING
        if x is None or y is None:
            for i in range(len(recent_frames) - 1, 0, -1):
                prev, prev_frame_counter, prev_ground = recent_frames[i -
                                                                      1], recent_frame_counters[i - 1], recent_ground[i - 1]
                anim, frame_counter, ground = recent_frames[i], recent_frame_counters[i], recent_ground[i]

                if self.is_new_action(prev, prev_frame_counter, anim, frame_counter) and self.__is_airborne(prev_ground, prev):
                    airborne_idx = i - 1
                    x, y = recent_coords[i - 1]
                    break
                elif i == (len(recent_frames) - 1) and self.is_new_action(prev, prev_frame_counter, anim, frame_counter) and self.__is_airborne(ground, anim):
                    airborne_idx = i
                    x, y = recent_coords[i]
                    break

        # 4. FINAL FALLBACK
        if x is None or y is None:
            x, y = recent_coords[-1]

        if airdodge_idx is not None:
            idx = airdodge_idx
        elif landing_idx is not None:
            idx = landing_idx
        elif airborne_idx is not None:
            idx = airborne_idx
        else:
            idx = len(recent_coords) - 1

        # resolve zero vector
        if np.isclose(x, 0.0) and np.isclose(y, 0.0):
            if idx > 0:
                x, y = recent_coords[idx - 1]
            elif len(recent_coords) >= 2:
                dx = recent_coords[-1][0] - recent_coords[-2][0]
                dy = recent_coords[-1][1] - recent_coords[-2][1]
                x, y = dx, dy

        # compute angle
        if not np.isclose(x, 0.0) or not np.isclose(y, 0.0):
            angle = math.atan2(abs(y), abs(x))
            
            return max(0.0, min(90.0, math.degrees(angle)))

        return None

    def __is_aerial_startup(self, anim):
        """Determine whether an animation is an aerial attack startup.

        Args:
            anim: The player's current animation.

        Returns:
            True if the animation is the startup of a neutral, forward,
            back, up, or down aerial attack.
        """

        return anim in (melee.Action.NAIR.value, melee.Action.FAIR.value, melee.Action.BAIR.value, melee.Action.UAIR.value, melee.Action.DAIR.value)

    def __get_aerial_attr(self, anim):
        """Get the attribute used to store frames for an aerial attack.

        Args:
            anim: The player's current animation.

        Returns:
            The name of the corresponding aerial frame attribute, or None
            if the animation is not an aerial attack.
        """

        if anim == melee.Action.NAIR.value:
            return "nair_frames"
        if anim == melee.Action.FAIR.value:
            return "fair_frames"
        if anim == melee.Action.BAIR.value:
            return "bair_frames"
        if anim == melee.Action.UAIR.value:
            return "uair_frames"
        if anim == melee.Action.DAIR.value:
            return "dair_frames"
        
        return None

    def is_new_action(self, previous_anim, previous_frame_counter, current_anim, current_frame_counter):
        """Determine whether the current frame starts a new action.

        An action is considered new when the animation changes or when the
        animation frame counter decreases, indicating that the animation
        has looped or restarted.

        Args:
            previous_anim: The animation on the previous frame.
            previous_frame_counter: The animation frame counter on the
                previous frame.
            current_anim: The animation on the current frame.
            current_frame_counter: The animation frame counter on the
                current frame.

        Returns:
            True if the current frame represents the start of a new action.

        Credit: Fizzi/Slippi
        """

        return (current_anim != previous_anim) or (previous_frame_counter > current_frame_counter)

    def track_dash_dances(self):
        """Track a completed dash dance from the player's recent actions.
        
        A dash dance is recorded when the most recent frames match the
        expected dash-dance pattern. The method identifies the beginning of
        the dash sequence, then records its duration and total displacement.

        Updates:
            Increments ``dash_dances`` and appends the dash dance's duration
            and displacement to ``dash_dance_lengths`` and
            ``dash_dance_distances``, respectively.
        """

        LOOKBACK = DASH_DANCE_FRAMES

        if len(self.animations) < LOOKBACK:
            return

        f = len(self.animations) - 1
        last3 = self.animations[-LOOKBACK:]

        if not self.__is_dash_dance_frame(last3):
            return

        start = None
        i = f - 2

        while i >= 1:
            if self.animations[i] == melee.Action.DASHING.value and self.is_new_action(self.animations[i - 1], self.frame_counters[i - 1], self.animations[i], self.frame_counters[i]):
                start = i
                break
            i -= 1

        if start is None:
            start = f - 2

        end = f

        length = end - start + 1
        distance = math.sqrt((self.positions_x[end] - self.positions_x[start])**2 + (self.positions_y[end] - self.positions_y[start])**2)

        self.dash_dances += 1
        self.dash_dance_lengths.append(length)
        self.dash_dance_distances.append(distance)

    def track_wavedashes(self):
        """Track a wavedash or waveland from the player's recent actions.

        A wavedash is identified when a special landing follows an animation
        capable of initiating a wavedash. The method uses the surrounding
        action sequence, airborne duration, vertical displacement, and
        controller input to distinguish wavedashes from wavelands and other
        special landings.

        Updates:
            Increments ``wavedash_count`` and records the wavedash duration
            and angle when a wavedash is detected. Increments
            ``waveland_count`` when the sequence is classified as a waveland.

        Credit: Fizzi/Slippi
        """

        if len(self.animations) < 2:
            return

        prev, current = self.animations[-2], self.animations[-1]

        is_special_land = current == melee.Action.LANDING_SPECIAL.value
        is_accepting_prev = self.__is_wavedash_initiation_animation(prev)
        is_possible_wavedash = is_special_land and is_accepting_prev

        if not is_possible_wavedash:
            return

        LOOKBACK = WAVEDASH_FRAMES

        recent_frames = self.animations[-LOOKBACK:]
        recent_frame_counters = self.frame_counters[-LOOKBACK:]
        recent_coords = self.processed_coords[-LOOKBACK:]
        recent_ground = self.on_ground[-LOOKBACK:]
        recent_l_digital = self.l_digital[-LOOKBACK:]
        recent_r_digital = self.r_digital[-LOOKBACK:]
        recent_set = set(recent_frames)

        if len(recent_set) == 2 and melee.Action.AIRDODGE.value in recent_set:
            return

        if melee.Action.KNEE_BEND.value in recent_set:
            airborne_frames = self.__count_airborne_frames(recent_frames, recent_ground)
            recent_y = self.positions_y[-LOOKBACK:]
            current_idx, knee_idx = len(recent_frames) - 1, None

            for i in range(current_idx, -1, -1):
                if recent_frames[i] == melee.Action.KNEE_BEND.value:
                    knee_idx = i
                    break

            y_diff = recent_y[current_idx] - recent_y[knee_idx]
            changed_y = abs(y_diff) > 0.1

            if airborne_frames >= 5 and changed_y:
                self.waveland_count += 1
            else:
                wavedash_frame = current_idx - knee_idx

                angle_deg = self.__compute_angle_from_recent(
                    recent_frames,
                    recent_frame_counters,
                    recent_coords,
                    recent_ground,
                    recent_l_digital,
                    recent_r_digital,
                )

                self.wavedash_count += 1
                self.wavedash_frames.append(wavedash_frame)
                self.wavedash_angles.append(round(angle_deg, 4) if angle_deg is not None else angle_deg)
        else:
            self.waveland_count += 1

    def track_sdi_windows(self):
        """Track an SDI opportunity and determine whether SDI occurred.

        A hitlag window is treated as an SDI opportunity. Controller inputs
        within the window are classified into directional regions, and valid
        transitions between regions are counted as SDI inputs. When at least
        one valid transition occurs, the window is recorded as an SDI move
        along with the number and combined magnitude of its inputs.

        Updates:
            Increments ``sdi_opportunities`` for each detected hitlag window.
            When SDI occurs, increments ``sdi_moves`` and appends the number
            of inputs and their combined magnitude to ``sdi_inputs`` and
            ``sdi_magnitudes``. Updates ``sdi_proportion`` after each
            opportunity.
        """

        if len(self.processed_coords) < 3:
            return

        i = len(self.processed_coords) - 1

        if self.hitlag_left[i - 1] > 0 and self.hitlag_left[i] <= 0:
            end = i - 1
            start = end

            while start > 0 and self.hitlag_left[start - 1] > 0:
                start -= 1

            if end - start < 1:
                return

            last_region = None
            inputs = 0
            vx = vy = 0.0

            for j in range(start, end + 1):  # lower bound might be start + 1
                x, y = self.processed_coords[j]
                region = self.__get_sdi_region(x, y)

                if (last_region is not None) and self.__is_valid_sdi_transition(last_region, region):
                    inputs += 1
                    vx += x
                    vy += y

                last_region = region

            magnitude = math.sqrt(vx * vx + vy * vy)

            if inputs > 0:
                self.sdi_moves += 1
                self.sdi_inputs.append(inputs)
                self.sdi_magnitudes.append(magnitude)

            self.sdi_opportunities += 1
            self.sdi_proportion = self.sdi_moves / self.sdi_opportunities

    def track_early_aerials(self):
        """Track the number of frames between a jump and an aerial attack.

        An aerial is recorded when a new aerial attack animation begins.
        The method searches the preceding frames for the start of the
        player's jump and records the number of frames between the jump
        startup and aerial startup.

        Updates:
            Appends the measured startup duration to the attribute
            corresponding to the aerial type (``nair_frames``,
            ``fair_frames``, ``bair_frames``, ``uair_frames``, or
            ``dair_frames``).
        """

        if len(self.animations) < 2:
            return

        current, current_frame_counter = self.animations[-1], self.frame_counters[-1]
        previous, previous_frame_counter = self.animations[-2], self.frame_counters[-2]

        if not (self.__is_aerial_startup(current) and self.is_new_action(previous, previous_frame_counter, current, current_frame_counter)):
            return

        LOOKBACK = EARLY_AERIAL_FRAMES
        frames_since_jump = self.__count_frames_since(LOOKBACK, melee.Action.KNEE_BEND.value)

        if frames_since_jump is None:
            return

        attr = self.__get_aerial_attr(current)
        if attr is not None:
            getattr(self, attr).append(frames_since_jump)

    def track_uptilts(self):
        """Track up-tilts performed from crouching or a pivot.

        A new up-tilt is classified as either a crouch up-tilt or pivot
        up-tilt based on the most recent qualifying crouch and pivot events.
        The number of frames since the corresponding event is recorded.

        Updates:
            Increments either ``crouch_uptilt_count`` or
            ``pivot_uptilt_count`` and appends the corresponding frame
            count to ``crouch_uptilt_frames`` or ``pivot_uptilt_frames``.
        """

        if len(self.animations) < 2:
            return
        
        current, current_frame_counter = self.animations[-1], self.frame_counters[-1]
        previous, previous_frame_counter = self.animations[-2], self.frame_counters[-2]

        if not (current == melee.Action.UPTILT.value and self.is_new_action(previous, previous_frame_counter, current, current_frame_counter)):
            return

        C_LOOKBACK, P_LOOKBACK = CROUCH_UPTILT_FRAMES, PIVOT_LOOKBACK
        crouch_frames, pivot_frames = self.animations[-C_LOOKBACK:], self.animations[-P_LOOKBACK:]
        has_crouch = any(self.__is_crouch(anim) for anim in crouch_frames)
        has_pivot = any(
            self.__is_pivot_frame(pivot_frames[i:i + PIVOT_FRAMES])
            for i in range(len(pivot_frames) - PIVOT_FRAMES + 1)
        )

        if not (has_crouch or has_pivot):
            return
        
        crouch_animations = [melee.Action.CROUCH_START.value, melee.Action.CROUCHING.value, melee.Action.CROUCH_END.value]

        frames_since_crouch_animations = [self.__count_frames_since(C_LOOKBACK, anim) for anim in crouch_animations]
        frames_since_pivot = self.__count_frames_since_pivot(pivot_frames)

        all_frames_since_crouch_none = all(fs is None for fs in frames_since_crouch_animations)
        all_frames_since_pivot_none = frames_since_pivot is None

        if all_frames_since_crouch_none and all_frames_since_pivot_none:
            return
        
        if all_frames_since_pivot_none:
            frames_since_crouch = min([fs for fs in frames_since_crouch_animations if fs is not None])

            self.crouch_uptilt_count += 1
            self.crouch_uptilt_frames.append(frames_since_crouch)
        elif all_frames_since_crouch_none:
            self.pivot_uptilt_count += 1
            self.pivot_uptilt_frames.append(frames_since_pivot)
        else:
            frames_since_crouch = min([fs for fs in frames_since_crouch_animations if fs is not None])

            if frames_since_crouch < frames_since_pivot:
                self.crouch_uptilt_count += 1
                self.crouch_uptilt_frames.append(frames_since_crouch)
            else:
                self.pivot_uptilt_count += 1
                self.pivot_uptilt_frames.append(frames_since_pivot)

    def track_forwardtilts(self):
        """Track forward tilts performed from a pivot.

        A new forward tilt is recorded when a qualifying pivot is detected
        within the lookback window. The number of frames between the pivot
        and forward tilt is recorded.

        Updates:
            Increments ``pivot_forwardtilt_count`` and appends the number
            of frames since the pivot to ``pivot_forwardtilt_frames``.
        """

        if len(self.animations) < 2:
            return
        
        current, current_frame_counter = self.animations[-1], self.frame_counters[-1]
        previous, previous_frame_counter = self.animations[-2], self.frame_counters[-2]

        if not (melee.Action.FTILT_HIGH.value <= current <= melee.Action.FTILT_LOW.value and self.is_new_action(previous, previous_frame_counter, current, current_frame_counter)):
            return
        
        LOOKBACK = PIVOT_LOOKBACK

        recent_frames = self.animations[-LOOKBACK:]

        has_pivot = any(
            self.__is_pivot_frame(recent_frames[i:i + PIVOT_FRAMES])
            for i in range(len(recent_frames) - PIVOT_FRAMES + 1)
        )

        if not has_pivot:        
            return
        
        frames_since_pivot = self.__count_frames_since_pivot(recent_frames)

        if frames_since_pivot is None:
            return

        self.pivot_forwardtilt_count += 1
        self.pivot_forwardtilt_frames.append(frames_since_pivot)

    def track_downtilts(self):
        """Track down tilts performed from a pivot.

        A new down tilt is recorded when a qualifying pivot is detected
        within the lookback window. The number of frames between the pivot
        and down tilt is recorded.

        Updates:
            Increments ``pivot_downtilt_count`` and appends the number
            of frames since the pivot to ``pivot_downtilt_frames``.
        """

        if len(self.animations) < 2:
            return
        
        current, current_frame_counter = self.animations[-1], self.frame_counters[-1]
        previous, previous_frame_counter = self.animations[-2], self.frame_counters[-2]

        if not (current == melee.Action.DOWNTILT.value and self.is_new_action(previous, previous_frame_counter, current, current_frame_counter)):
            return
        
        LOOKBACK = PIVOT_LOOKBACK

        recent_frames = self.animations[-LOOKBACK:]

        has_pivot = any(
            self.__is_pivot_frame(recent_frames[i:i + PIVOT_FRAMES])
            for i in range(len(recent_frames) - PIVOT_FRAMES + 1)
        )

        if not has_pivot:        
            return
        
        frames_since_pivot = self.__count_frames_since_pivot(recent_frames)

        if frames_since_pivot is None:
            return

        self.pivot_downtilt_count += 1
        self.pivot_downtilt_frames.append(frames_since_pivot)

    def track_tilts(self):
        """Track up, forward, and down tilts.

        Delegates tilt detection to ``track_uptilts``,
        ``track_forwardtilts``, and ``track_downtilts``.
        """

        self.track_uptilts()
        self.track_forwardtilts()
        self.track_downtilts()

    def track_jc_shines(self):
        """Track jump-canceled shines and their jump-to-shine timing.

        A jump-canceled shine is recorded when a new shine animation begins
        within the configured lookback period following jump startup. The
        number of frames between jump startup and the shine is recorded.

        Updates:
            Increments ``jc_shine_count`` and appends the number of frames
            since jump startup to ``jc_shine_frames``.
        """

        if len(self.animations) < 2:
            return
        
        current, current_frame_counter = self.animations[-1], self.frame_counters[-1]
        previous, previous_frame_counter = self.animations[-2], self.frame_counters[-2]

        if not (self.__is_shine(current) and self.is_new_action(previous, previous_frame_counter, current, current_frame_counter)):
            return
        elif current != melee.Action.DOWN_B_STUN.value:
            return

        LOOKBACK = JC_SHINE_FRAMES
        frames_since_jump = self.__count_frames_since(LOOKBACK, melee.Action.KNEE_BEND.value)

        if frames_since_jump is None:
            return
        
        self.jc_shine_count += 1
        self.jc_shine_frames.append(frames_since_jump)

    def count_ledgedashes(self):
        """Analyze ledge escapes and track ledgedash-related metrics.

        The method scans the player's action history to identify ledge
        releases, track the resulting galint window, and classify outcomes
        following the ledge release. Successful ledgedashes are recorded
        along with their input angles, while air-dodge self-destructs and
        no-impact landings are tracked separately.

        The method also calculates the maximum and mean galint observed
        across successful ledgedashes and the proportion of ledgedash
        attempts that result in an air-dodge self-destruct.

        Updates:
            Sets ``intangible_ledgedashes`` to the number of successful
            ledgedashes that reach an intangible landing.

            Sets ``ledgedash_angles`` to the input angles associated with
            successful ledgedashes.

            Sets ``ledgedash_sd_proportion`` to the proportion of
            ledgedash attempts resulting in an air-dodge self-destruct.

            Sets ``no_impact_lands`` to the number of no-impact landings
            following a ledge release.

            Sets ``galint`` to the galint values recorded for successful
            ledgedashes.

            Sets ``max_galint`` and ``mean_galint`` to the maximum and mean
            galint values, respectively.

        Credit: cbartsch/slippic
        """

        galint = 0
        offledge = False

        intangible_ledgedashes = 0
        ledgedash_angles = []
        no_impact_lands = 0
        sds = 0
        max_galint = 0
        galint_list = []

        f = 1
        frame_count = len(self.animations)

        while f < frame_count:
            prev_pf, pf = self.animations[f - 1], self.animations[f]

            if galint > 0:
                galint -= 1

                if not offledge:
                    offledge = self.__did_release_ledge(prev_pf, pf)
                    f += 1
                    continue
                
                airdodge_lookahead = min(f + FRAMES_PER_SECOND + 1, frame_count)
                is_airdodge = (pf == melee.Action.AIRDODGE.value) and self.is_new_action(prev_pf, self.frame_counters[f - 1], pf, self.frame_counters[f])
                does_airdodge_sd = self.__is_airdodge_sd(self.animations[f:airdodge_lookahead])

                landed = (self.__is_landing(prev_pf) and (not self.__is_landing(pf)) and (not self.__is_airborne(self.on_ground[f], pf)))
                no_impact_land = self.__did_no_impact_land(prev_pf, pf)

                if (no_impact_land or landed) and (self.hitlag_left[f] <= 0) and (self.hitstun_frames_left[f] <= 0):
                    if no_impact_land:
                        no_impact_lands += 1
                    else:
                        LOOKBACK = LEDGEDASH_FRAMES

                        start_idx = max(0, f - LOOKBACK)
                        recent_frames = self.animations[start_idx:f+1]
                        recent_frame_counters = self.frame_counters[start_idx:f+1]
                        recent_coords = self.processed_coords[start_idx:f+1]
                        recent_ground = self.on_ground[start_idx:f+1]
                        recent_l_digital = self.l_digital[start_idx:f+1]
                        recent_r_digital = self.r_digital[start_idx:f+1]

                        angle_deg = self.__compute_angle_from_recent(
                            recent_frames,
                            recent_frame_counters,
                            recent_coords,
                            recent_ground,
                            recent_l_digital,
                            recent_r_digital,
                        )

                        ledgedash_angles.append(
                            round(angle_deg, 4) if angle_deg is not None else angle_deg)
                        intangible_ledgedashes += 1

                    max_galint = max(max_galint, galint)
                    galint_list.append(galint)

                    f += galint + 1
                    galint = 0
                    offledge = False
                    continue

                elif is_airdodge and does_airdodge_sd:
                    sds += 1

                    f += airdodge_lookahead + 1
                    galint = 0
                    offledge = False
                    continue

            if self.__did_cliff_catch_end(prev_pf, pf):
                galint = 30

                if self.__did_release_ledge(prev_pf, pf):
                    offledge = True

            f += 1

        total_ledgedash_attempts = sds + intangible_ledgedashes
        mean_galint = np.mean(galint_list) if len(galint_list) > 0 else 0.0
        ledgedash_sd_proportion = (sds / total_ledgedash_attempts) if total_ledgedash_attempts > 0 else 0.0

        self.intangible_ledgedashes = intangible_ledgedashes
        self.ledgedash_angles = ledgedash_angles
        self.ledgedash_sd_proportion = ledgedash_sd_proportion
        self.no_impact_lands = no_impact_lands
        self.galint = galint_list
        self.max_galint = max_galint
        self.mean_galint = mean_galint

    def finalize(self):
        """Finalize player-state metrics that require complete replay data.

        Performs the final ledgedash analysis after all player frames have
        been processed.
        """

        self.count_ledgedashes()
