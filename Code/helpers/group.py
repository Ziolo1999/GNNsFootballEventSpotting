import pandas as pd
import logging
import numpy as np

def base_phases(match: pd.DataFrame) -> pd.DataFrame:
    """Generated the most basic form a phase
    Uses ball ownership and game state (alive or dead) to split the game into phases

    Args:
        match (pd.DataFrame): match(es) to split into phases

    Returns:
        pd.DataFrame: df with added col that is True in the rows where a phase starts
    """
    # make a copy and add the "index" as its own column
    # two reset index necessary to make the index col correspond to the correct index 
    match = match.reset_index().drop(columns=["index"]).reset_index()

    # easily detect ball owner changes
    match["owner_role_prev"] = match["ball_owner_v3_role"].shift(1).fillna(0)
    match["owner_prev"] = match["ball_owner"].shift(1).fillna(0)
    match["info_matchnr_prev"] = match["info_matchnr"].shift(1).fillna(0) # first match has index 0
    match["info_period_prev"] = match["info_period"].shift(1).fillna(0) # first match has index 0

    # phase start when gaining/losing possession
    # OR when new match starts!
    # OR when new period starts!
    # OR when the ball was dead (foul, out of bounds etc..)
    match["phase_start"] = (match["owner_prev"] != match["ball_owner"]) |\
        (match["info_matchnr"] != match["info_matchnr_prev"])  |\
        (match["info_period"] != match["info_period_prev"])  |\
        (match["info_was_dead"])
    
    return match

def find_ball_unstable(match: pd.DataFrame, threshold=40) -> list:
    """Return indices of phase starts that are started during high ball contention.
    Phases started in this time are combined with the previous phase until no rapid ball ownership changes occur

    Args:
        match (pd.DataFrame): the match
        threshold (int, optional): frame interval in which ball contention may occur. Defaults to 40.

    Returns:
        list: list of phase starts that should be cancelled
    """
    indices = []
    prev_phase = 0
    time_since_ball_switch = 0

    dead_start = True
    period_start = True
    match_start = True

    for index, row in match.iterrows():
        if row["phase_start"]:
            if not dead_start and not period_start and not match_start:
                # assert row["ball_owner"] != row["owner_prev"], "cause of period start unknown"
                if time_since_ball_switch < threshold:
                    indices.append(prev_phase)

            dead_start = row["info_was_dead"]
            period_start = row["info_period"] != row["info_period_prev"]
            match_start = row["info_matchnr"] != row["info_matchnr_prev"]
            prev_phase = index

        if row["ball_owner"] != row["owner_prev"]:
            time_since_ball_switch = 0

        time_since_ball_switch += 1

    return indices

def combine_phases(match: pd.DataFrame) -> list:
    """After removing phases started because of high ball contention, two subsequent phases may have ball possession on 
    the same team. This should still be the same phase. This function finds those phases.

    Args:
        match (pd.DataFrame): the match

    Returns:
        list: phase starts that should be cancelled
    """
    indices = []
    subset = match.loc[match["phase_start"]]
    prev_ball_owner = None
    for index, row in subset.iterrows():
        ball_owner = row["ball_owner"]
        if ball_owner == prev_ball_owner:
            # same phase if not dead ball or period switch or match switch
            if row["info_matchnr"] == row["info_matchnr_prev"] and row["info_period"] == row["info_period_prev"] and \
                not row["info_was_dead"]:
                # phase seperation was caused by ball switch
                indices.append(index)
        
        prev_ball_owner = ball_owner

    return indices

def find_slowdown(match: pd.DataFrame, line_diff=0.05, ball_dist_travelled=0.3, speed_threshold=0.2,
        keeper_phase_length=40, keeper_ball_dist_travelled=0.3):
    """Detect when during a phase, progress slows. Start a new phase if this happens.
    Progress slows iff the enemy team is back in position and the ball is in front of the enemy line (top 3 enemy players)
    Also created new phases when passed to the keeper (under certain conditions)

    Args:
        match (pd.DataFrame): the match
        line_diff (float, optional): distance to the enemy line. Defaults to 0.05.
        ball_dist_travelled (float, optional): distance ball needs to travel before starting new phase. Defaults to 0.3.
        keeper_phase_length (int, optional): Min time between phases started by passing to keeper. Defaults to 0.3.
        keeper_ball_dist_travelled (float, optional): distance ball needs to travel before starting new phase. Defaults to 0.3.

    Returns:
        Tuple[list, list]: indices of new phases, ball distance travelled since phase start for each frame
    """
    match["enemy_stable"] = match["enemy_centroid_speed"] < speed_threshold
    spans = []
    new_starts = []
    phase_length = 0
    for index, row in match.iterrows():
        if row["phase_start"]:
            ball_start = row["ball_px"]
            spans.append(0)
            phase_length = 0
            
        else:
            current_span = row["ball_px"] - ball_start
            max_span = max(current_span, spans[-1])
            phase_length += 1

            # flow detection
            if row["enemy_stable"] and row["enemy_line"] - line_diff > row["ball_px"] and max_span > ball_dist_travelled:
                new_starts.append(index)
                ball_start = row["ball_px"]
                spans.append(0)

            # keeper detection
            elif row["owner_role_prev"] > 0 and row["ball_owner_v3_role"] == 0:
                # phase was long enough, progress was made
                if phase_length > keeper_phase_length and max_span > keeper_ball_dist_travelled:
                    new_starts.append(index)
                    spans.append(0)

            else:
                spans.append(max_span)

    return new_starts, spans

def find_slowdown2(match: pd.DataFrame, line_diff=0.05, ball_dist_travelled=0.3, speed_threshold=0.2,
        keeper_phase_length=40, keeper_ball_dist_travelled=0.3):
    """Detect when during a phase, progress slows. Start a new phase if this happens.
    Progress slows iff the enemy team is back in position and the ball is in front of the enemy line (top 3 enemy players)
    Also created new phases when passed to the keeper (under certain conditions)
    added in v2: if ball starts @0.9 and then goes to 0.6 no new phase was started in previous versions, v2 fixes this.

    Args:
        match (pd.DataFrame): the match
        line_diff (float, optional): distance to the enemy line. Defaults to 0.05.
        ball_dist_travelled (float, optional): distance ball needs to travel before starting new phase. Defaults to 0.3.
        keeper_phase_length (int, optional): Min time between phases started by passing to keeper. Defaults to 0.3.
        keeper_ball_dist_travelled (float, optional): distance ball needs to travel before starting new phase. Defaults to 0.3.

    Returns:
        Tuple[list, list, list]: indices of new phases, ball distance travelled since phase start for each frame forward & backwards
    """
    match["enemy_stable"] = match["enemy_centroid_speed"] < speed_threshold
    spans_forward = []
    spans_backward = []
    new_starts = []
    phase_length = 0
    for index, row in match.iterrows():
        # check for out of bounds ball
        ball_px = max(row["ball_px"], 0)
        ball_px = min(ball_px, 1)

        if row["phase_start"]:
            ball_start = ball_px
            spans_forward.append(0)
            spans_backward.append(0)
            phase_length = 0
            
        else:
            phase_length += 1
            current_span = ball_px- ball_start
            current_span_edit = min(abs(current_span), abs(ball_px - 0.3)) # prevent backwards flow by pass to keeper, detect that seperately
            max_span = max(current_span, spans_forward[-1])
            min_span = max(current_span_edit, spans_backward[-1])

            # flow detection towards goal
            flow_forward_stopped = row["enemy_stable"] and row["enemy_line"] - line_diff > ball_px and max_span > ball_dist_travelled
            # flow detection towards goal
            flow_backward_stopped = row["enemy_stable"] and row["enemy_line"] - line_diff > ball_px and abs(min_span) > ball_dist_travelled 
            if flow_forward_stopped or flow_backward_stopped:
                new_starts.append(index)
                ball_start = ball_px
                spans_forward.append(0)
                spans_backward.append(0)

            # keeper detection
            elif row["owner_role_prev"] > 0 and row["ball_owner_v3_role"] == 0:
                # phase was long enough, progress was made
                if phase_length > keeper_phase_length and max_span > keeper_ball_dist_travelled:
                    new_starts.append(index)
                    ball_start = ball_px
                    spans_forward.append(0)
                    spans_backward.append(0)

            else:
                spans_forward.append(max_span)
                spans_backward.append(min_span)

    return new_starts, spans_forward, spans_backward


def group(match: pd.DataFrame, features: list[str], **kwargs) -> pd.DataFrame:
    """Group the values of a match into phases by grouping them on their phase number.
    Statistical or sequential representation of the features are returned.

    Args:
        match (pd.DataFrame): match(es) to group
        features (list[str]): features to keep

    Returns:
        pd.DataFrame: df with a row for each phase
    """
    # columns needed to groupby
    base_cols = ["phase_start", "owner_prev", "ball_owner", "owner_role_prev", "ball_owner_v3_role", "index", "phase_time", "info_matchnr", "info_time", "info_period"]
    group_cols = features + base_cols 
    feature_group = {
        key: ["first", "last", "min", "max", "mean", "std"] for key in features
    }
    group_type = kwargs.get("group_type", None)
    if group_type == "bins":
        def to_bins(time_sequence, bins):
            l = time_sequence.shape[0]
            if l < 10:
                return {f"{time_sequence.name}{i}": 0 for i in range(10)}

            lengths = [l // 10 for _ in range(10)]
            remainder = l % bins
            for i in range(remainder):
                lengths[i] += 1

            averages = {}
            cur_index = 0
            for i, length in enumerate(lengths):
                averages[f"{time_sequence.name}{i}"] = np.average(time_sequence[cur_index:cur_index + length])
                cur_index += length

            return averages

        bins = kwargs.get("group_bins", 10)
        feature_group = {
            key: lambda x: to_bins(x, bins) for key in features
        }

    # create phase df
    grouped_on_phase = match[group_cols].groupby((match["phase_start"] == 1).cumsum()).agg({
        "index": ["first", "last"],
        "phase_start": "first",
        "ball_owner": "first",
        "owner_prev": "first",
        "ball_owner_v3_role": "first",
        "owner_role_prev": "first",
        "info_matchnr": "first",
        "info_time": ["first", "last"],
        "phase_time": "last",
        "info_period": "first",
        **feature_group
    })
    # rename columns (remove "first" from the names)
    grouped_on_phase.columns = [f'{i}_{j}' if j != 'first' else f'{i}' for i,j in grouped_on_phase.columns]

    # create span columns
    if group_type is None:
        for feature in features:
            grouped_on_phase[f"{feature}_span"] = grouped_on_phase[f"{feature}_max"] - grouped_on_phase[f"{feature}_min"]
    elif group_type == "bins":
        lambda_cols = [col for col in grouped_on_phase.columns if "<lambda>" in col]
        other_cols = set(grouped_on_phase.columns) - set(lambda_cols)

        dfs = [grouped_on_phase.loc[:, other_cols]]
        for col in lambda_cols:
            new_df = grouped_on_phase[col].apply(pd.Series)
            dfs.append(new_df)

        grouped_on_phase = pd.concat(dfs, axis=1)


    # make sure std is not NaN
    for col in grouped_on_phase.columns.tolist():
        if "_std" in col:
            grouped_on_phase[col].fillna(0, inplace=True)

    return grouped_on_phase

def group_on_phase(match: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Transform the home df dataset. Groupes data into "phases". A phase is a continuous sequence of ballownership,
    stopped when they lose ball possession or when they pass back to the keeper.
    Only uses the most basic version of phases (no flow detection)

    Args:
        features (list[str]): features columns 

    Returns:
        pd.DataFrame: new df with data grouped into phases
    """
    # make a copy and add the "index" as its own column
    # two reset index necessary to make the index col correspond to the correct index 
    match = match.reset_index().drop(columns=["index"]).reset_index()

    # easily detect ball owner changes
    owner_role_prev = pd.Series(match["ball_owner_v3_role"].shift(1).fillna(0), name="owner_role_prev")
    owner_prev = pd.Series(match["ball_owner"].shift(1).fillna(0), name="owner_prev")
    info_matchnr_prev = pd.Series(match["info_matchnr"].shift(1).fillna(0), name="info_matchnr_prev") # first match has index 0
    info_period_prev = pd.Series(match["info_period"].shift(1).fillna(0), name="info_period_prev") # first match has index 0
    match = pd.concat((match, owner_role_prev, owner_prev, info_matchnr_prev, info_period_prev), axis=1)

    # phase start when gaining/losing possession
    # OR when getting a pass from the keeper
    # OR when new match starts!
    # OR when new period starts!
    # OR when the ball was dead (foul, out of bounds etc..)
    match["phase_start"] = (match["owner_prev"] != match["ball_owner"]) |\
        ((match["owner_role_prev"] == 0) & (match["ball_owner_v3_role"] > 0)) |\
        (match["info_matchnr"] != match["info_matchnr_prev"])  |\
        (match["info_period"] != match["info_period_prev"])  |\
        (match["info_was_dead"])

    # TODO remove frames that are dead from grouping!

    # length of the phase
    match["phase_time"] = match.groupby((match["phase_start"] == 1).cumsum()).cumcount()
    # index of the phase
    match["phase_order"] = (match["phase_start"] == 1).cumsum()

    grouped_on_phase = group(match, features)
    return grouped_on_phase, match

def group_on_phases_v2(matches: pd.DataFrame, features: list[str], **kwargs):
    """Transform the matches dataframe into phases.
    Depending on kwarg slowdown_version backwards travel is accounted for 

    Args:
        matches (pd.DataFrame): match(es) to group into phases
        features (list[str]): features to keep

    Raises:
        Exception: if slowdown version is not implemented

    Returns:
        Tuple[pd.DataFrame, pd.Dataframe]: new phase df, matches with added phase info
    """
    print(kwargs)
    ball_interval = kwargs.get("ball_interval", 40)
    ball_dist_travelled = kwargs.get("ball_dist_travelled", 0.3)
    line_diff = kwargs.get("line_diff", 0.05)
    speed_threshold = kwargs.get("speed_threshold", 0.2)
    keeper_phase_length = kwargs.get("keeper_phase_length", 40)
    keeper_ball_dist_travelled = kwargs.get("keeper_ball_dist_travelled", 0.3)
    slowdown_version = kwargs.get("slowdown_version", 1)
    subset = kwargs.get('subset', None)

    if subset is not None:
        matches = matches.loc[matches["info_matchnr"].isin(subset)]
        logging.info(f"Matches subset {matches['info_matchnr'].unique()}")

    matches = base_phases(matches)
    # ball stableness
    to_remove = find_ball_unstable(matches, ball_interval)
    matches.loc[to_remove, "phase_start"] = False
    to_remove = combine_phases(matches)
    matches.loc[to_remove, "phase_start"] = False
    matches.loc[matches["phase_start"], "phase_reason"] = "base"

    # flow detection
    if slowdown_version == 1:
        to_add, _ = find_slowdown(matches, line_diff, ball_dist_travelled, speed_threshold, keeper_phase_length, keeper_ball_dist_travelled)
    elif slowdown_version == 2:
        to_add, _, _ = find_slowdown2(matches, line_diff, ball_dist_travelled, speed_threshold, keeper_phase_length, keeper_ball_dist_travelled)
    else:
        raise Exception("Unimplemented slowdown version")
    matches.loc[to_add, "phase_start"] = True
    matches.loc[to_add, "phase_reason"] = "flow"

    # length of the phase
    matches["phase_time"] = matches.groupby((matches["phase_start"] == 1).cumsum()).cumcount()
    # index of the phase
    matches["phase_order"] = (matches["phase_start"] == 1).cumsum()

    grouped_on_phase = group(matches, features, **kwargs)
    return grouped_on_phase, matches

def group_on_phases_v3(matches: pd.DataFrame, features: list[str], **kwargs):
    """Like group_on_phase_v2, but always uses newer slowdown version

    Args:
        matches (pd.DataFrame): match(es) to group into phases
        features (list[str]): features to keep

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: new phase df, matches with added phase info
    """
    kwargs["slowdown_version"] = 2
    return group_on_phases_v2(matches, features, **kwargs)