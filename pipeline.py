"""Rule-based possession labelling for BAR-ATH (tactical video).

Run:  python pipeline.py [output.csv]
Writes 199,319 rows `frame,label` (default: evaluation/results/my_team.csv).

Method:
  1. Load tracking, convert 0-1 coords to metres.
  2. Interpolate short ball gaps (<= 1 s).
  3. Raw signal: team of the nearest player to the ball, if within MAX_DIST m.
  4. Dead ball: ball visibly outside the pitch for a while (+ pre-kickoff 2nd half).
  5. Hold filter: possession only changes after the other team has been the
     nearest for MIN_HOLD consecutive frames. After a dead ball, the first
     confirmed team becomes the possessor.
"""
import sys
import numpy as np
import pandas as pd
from kloppy import metrica
from games import asset_paths  # provided by the starter repo

PITCH_L, PITCH_W = 105.3, 68.0
N_FRAMES = 199_319
HALVES = [(11_185, 90_651), (110_808, 190_197)]
SECOND_HALF_KICKOFF = 113_742

MAX_GAP = 12      # interpolate ball gaps up to 0.5s
MAX_DIST = 5.0    # metres: nearest player must be this close to count
MIN_HOLD = 50     # frames the other team must be nearest to flip possession
MIN_START = 25    # frames to confirm the first possessor after a restart
MIN_DEAD = 25     # frames outside the pitch before we call it dead


def load():
    paths = asset_paths()
    ds = metrica.load_tracking_epts(meta_data=paths["atd_meta"], raw_data=paths["atd_raw"])
    df = ds.to_df()
    home_ids, away_ids = set(), set()
    for team in ds.metadata.teams:
        ids = {str(p.player_id) for p in team.players}
        (home_ids if str(team.ground).lower().endswith("home") else away_ids).update(ids)
    return df, home_ids, away_ids


def team_columns(df, ids):
    cols = []
    for c in df.columns:
        if c.endswith("_x") and c[:-2] in ids:
            cols.append(c[:-2])
    return cols


def min_dist_to_ball(df, players, bx, by):
    if not players:
        return np.full(len(df), np.inf)
    px = df[[f"{p}_x" for p in players]].to_numpy() * PITCH_L
    py = df[[f"{p}_y" for p in players]].to_numpy() * PITCH_W
    d = np.hypot(px - bx[:, None], py - by[:, None])
    d = np.where(np.isnan(d), np.inf, d)
    return d.min(axis=1)


def runs_at_least(mask, n):
    """Keep only True-runs of length >= n."""
    out = np.zeros_like(mask)
    i, L = 0, len(mask)
    while i < L:
        if mask[i]:
            j = i
            while j < L and mask[j]:
                j += 1
            if j - i >= n:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def label_half(raw, dead, start, end):
    """raw: array of 'H'/'A'/'X' ; dead: bool array. Returns labels for [start, end]."""
    out = ["X"] * (end - start + 1)
    current, cand, count = None, None, 0
    for k in range(end - start + 1):
        f = start + k
        if dead[f]:
            out[k] = "D"
            current, cand, count = None, None, 0
            continue
        r = raw[f]
        if r in ("H", "A"):
            if current is None:
                if r == cand:
                    count += 1
                else:
                    cand, count = r, 1
                if count >= MIN_START:
                    current, cand, count = r, None, 0
            elif r != current:
                if r == cand:
                    count += 1
                else:
                    cand, count = r, 1
                if count >= MIN_HOLD:
                    current, cand, count = r, None, 0
            else:
                cand, count = None, 0
        out[k] = current if current else "X"
    return out


def main(out_path):
    df, home_ids, away_ids = load()
    home = team_columns(df, home_ids)
    away = team_columns(df, away_ids)
    print(f"home slots: {len(home)}, away slots: {len(away)}")
    if not home or not away:
        raise SystemExit("Team mapping failed: print df.columns[:10] and "
                         "ds.metadata.teams[0].players[:3] and adjust load()/team_columns().")

    df = df.copy()
    df["frame"] = df["frame_id"] - 1
    df = df.set_index("frame").reindex(range(N_FRAMES))

    # ball: interpolate short gaps, but remember where it was really out of bounds
    bx_raw, by_raw = df["ball_x"].to_numpy(), df["ball_y"].to_numpy()
    bx = df["ball_x"].interpolate(limit=MAX_GAP, limit_area="inside").to_numpy() * PITCH_L
    by = df["ball_y"].interpolate(limit=MAX_GAP, limit_area="inside").to_numpy() * PITCH_W

    dh = min_dist_to_ball(df, home, bx, by)
    da = min_dist_to_ball(df, away, bx, by)
    ball_ok = ~np.isnan(bx)
    raw = np.full(N_FRAMES, "X", dtype=object)
    near = ball_ok & (np.minimum(dh, da) <= MAX_DIST)
    raw[near & (dh < da)] = "H"
    raw[near & (da <= dh)] = "A"

    # dead ball: ball seen outside the pitch for >= MIN_DEAD frames
    outside = (~np.isnan(bx_raw)) & ((bx_raw < 0) | (bx_raw > 1) | (by_raw < 0) | (by_raw > 1))
    dead = runs_at_least(outside, MIN_DEAD)
    dead[HALVES[1][0]:SECOND_HALF_KICKOFF] = True  # known: graded but all dead

    labels = np.full(N_FRAMES, "X", dtype=object)
    for s, e in HALVES:
        labels[s:e + 1] = label_half(raw, dead, s, e)

    pd.DataFrame({"frame": np.arange(N_FRAMES), "label": labels}).to_csv(out_path, index=False)

    # sanity report
    for s, e in HALVES:
        seg = labels[s:e + 1]
        poss = int(((seg[1:] != seg[:-1]) & np.isin(seg[1:], ["H", "A"])).sum())
        print(f"half {s}-{e}: H={np.sum(seg=='H')} A={np.sum(seg=='A')} "
              f"D={np.sum(seg=='D')} X={np.sum(seg=='X')}  possession starts~{poss}")
    print("wrote", out_path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "evaluation/results/404era.csv")