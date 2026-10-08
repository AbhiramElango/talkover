from talkover.evaluation import TimeInterval, intersection_duration, merge_intervals, subtract_intervals


def _spans(*pairs):
    return tuple(TimeInterval(start, end) for start, end in pairs)


def test_merge_and_intersection() -> None:
    assert merge_intervals(_spans((0, 1), (0.5, 2), (3, 4))) == _spans((0, 2), (3, 4))
    assert intersection_duration(_spans((0, 2)), _spans((1, 3))) == 1


def test_subtract_removes_excluded_coverage() -> None:
    assert subtract_intervals(_spans((0, 4)), _spans((1, 2))) == _spans((0, 1), (2, 4))
