import math

from production.trajectory_compare import (
    MISSING_JOINT_DIST, _frame_dist, contact_landmarks, dtw_distance,
)

JOINTS = ['left_wrist', 'right_wrist']


def pose(x):
    return {'left_wrist': {'x': x, 'y': 0.0}, 'right_wrist': {'x': -x, 'y': 0.0}}


def traj(xs):
    return [{'t': i - len(xs) // 2, 'landmarks': pose(x)} for i, x in enumerate(xs)]


def test_frame_dist_averages_joints():
    assert _frame_dist(pose(0.0), pose(0.3), JOINTS) == 0.3


def test_joint_missing_on_one_side_is_penalised():
    a = {'left_wrist': {'x': 0.0, 'y': 0.0}}
    b = {'left_wrist': {'x': 0.0, 'y': 0.0}, 'right_wrist': {'x': 0.0, 'y': 0.0}}
    assert _frame_dist(a, b, JOINTS) == MISSING_JOINT_DIST / 2


def test_slow_copy_beats_different_swing():
    swing = [math.sin(i / 3) for i in range(12)]
    slow = [math.sin(i / 6) for i in range(24)]
    other = [math.cos(i / 3) for i in range(12)]
    assert dtw_distance(traj(swing), traj(swing), JOINTS) == 0
    assert dtw_distance(traj(swing), traj(slow), JOINTS) < dtw_distance(traj(swing), traj(other), JOINTS)


def test_empty_trajectory_is_infinitely_far():
    assert dtw_distance([], traj([0.0]), JOINTS) == float('inf')


def test_contact_landmarks_picks_t_zero():
    assert contact_landmarks(traj([0.1, 0.2, 0.3])) == pose(0.2)
