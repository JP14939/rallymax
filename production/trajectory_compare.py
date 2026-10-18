"""
Compare two pose trajectories (sequences of normalised landmark snapshots
across a swing) using Dynamic Time Warping — aligns sequences that differ in
length/speed, which a flattened fixed-length vector can't do. This replaces
comparing 3 fixed snapshots (backswing/contact/followthrough) with comparing
the full ~20fps motion path.
"""
import math

# MediaPipe's z (rough, hip-relative monocular depth) turned out to be
# on a MUCH wider scale than x/y in practice -- measured directly against a
# real rebuilt database: abs-median z ~1.83 (range -17.2 to 12.3) vs abs-
# median x ~0.60. The assumption that z is "the same normalized
# image-width units as x/y" does NOT hold empirically. A live before/after
# test on 4 real saved swings showed
# Z_WEIGHT=0.5 caused similarity scores to drop 45-75% on 3 of 4 clips --
# z was dominating the distance metric, not contributing a minor signal.
# DISABLED (0.0) until z is properly rescaled (e.g. normalize by its own
# measured spread, not just divided by shoulder width like x/y) and
# re-validated -- do not re-enable by just picking a smaller constant here
# without re-measuring z's actual distribution first.
Z_WEIGHT = 0.0

# A landmark present on one side only used to be skipped, so a trajectory missing its fastest joint (the hitting
# wrist) averaged over the slow ones and looked close to everything: it became a DTW hub (2026-10-13: one practice
# clip was the top match for 7/18 held-out pro serves). Now it costs this much, the p90 per-joint distance
# between queries and their top matches (median 0.33), i.e. "about as far as a bad joint". Missing on both
# sides is still skipped: neither side says anything about it.
MISSING_JOINT_DIST = 1.18


def _frame_dist(a, b, key_landmarks):
    """Average per-landmark euclidean distance between two landmark dicts.
    Includes z (down-weighted by Z_WEIGHT) when both sides have it; falls
    back to pure 2D for any landmark missing z on either side (e.g. an
    older cached trajectory built before z was carried through).
    A landmark on one side only counts MISSING_JOINT_DIST."""
    total = 0.0
    n = 0
    for name in key_landmarks:
        pa, pb = a.get(name), b.get(name)
        if pa is None and pb is None:
            continue
        if pa is None or pb is None:
            total += MISSING_JOINT_DIST
            n += 1
            continue
        dx = pa['x'] - pb['x']
        dy = pa['y'] - pb['y']
        za, zb = pa.get('z'), pb.get('z')
        if za is not None and zb is not None:
            dz = (za - zb) * Z_WEIGHT
            total += math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        else:
            total += math.sqrt(dx ** 2 + dy ** 2)
        n += 1
    if n == 0:
        return 1.5  # no shared landmarks this frame pair — treat as a poor match
    return total / n


def dtw_distance(traj_a, traj_b, key_landmarks):
    """
    DTW distance between two trajectories, normalised by path length so it's
    comparable across swings of different durations. Lower = more similar.
    """
    n, m = len(traj_a), len(traj_b)
    if n == 0 or m == 0:
        return float('inf')

    INF = float('inf')
    dtw = [[INF] * (m + 1) for _ in range(n + 1)]
    dtw[0][0] = 0.0

    for i in range(1, n + 1):
        li = traj_a[i - 1]['landmarks']
        row, prev_row = dtw[i], dtw[i - 1]
        for j in range(1, m + 1):
            cost = _frame_dist(li, traj_b[j - 1]['landmarks'], key_landmarks)
            row[j] = cost + min(prev_row[j], row[j - 1], prev_row[j - 1])

    return dtw[n][m] / max(n, m)


def contact_landmarks(trajectory):
    """The trajectory snapshot closest to t=0 (contact)."""
    if not trajectory:
        return {}
    return min(trajectory, key=lambda p: abs(p['t']))['landmarks']
