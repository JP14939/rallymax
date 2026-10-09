"""
Toy Dynamic Time Warping demo on synthetic 1-D "swings".

DTW aligns two sequences that play out at different speeds before comparing
them, so a slow swing and a fast swing of the same shape still match. This is
the textbook algorithm on made-up curves, not RallyMax's production pipeline.

    python demo/dtw_demo.py
"""
import math
import random


def dtw_distance(a, b):
    """Classic O(n*m) DTW, normalised by the longer sequence's length."""
    n, m = len(a), len(b)
    inf = float('inf')
    d = [[inf] * (m + 1) for _ in range(n + 1)]
    d[0][0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(a[i - 1] - b[j - 1])
            d[i][j] = cost + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])
    return d[n][m] / max(n, m)


def swing(frames=40):
    """A made-up swing: wind back, accelerate through, follow through."""
    return [math.sin(math.pi * t / (frames - 1)) ** 2 * (1 - t / frames) for t in range(frames)]


def other_swing(frames=40):
    return [math.cos(2 * math.pi * t / (frames - 1)) * 0.5 for t in range(frames)]


def cases():
    rng = random.Random(0)
    ref = swing()
    return ref, [
        ('same swing', ref),
        ('same swing at half speed', swing(80)),
        ('same swing with noise', [x + rng.gauss(0, 0.03) for x in ref]),
        ('a different swing', other_swing()),
    ]


def main():
    ref, rows = cases()
    print('DTW distance from the reference swing (lower = more similar)\n')
    for label, seq in rows:
        print(f'  {label:<26} {dtw_distance(ref, seq):.3f}')


if __name__ == '__main__':
    main()
