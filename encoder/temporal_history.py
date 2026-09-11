"""Episode-local causal frame indices; no pretraining targets or losses."""

def causal_history_indices(anchor: int, sequence_length: int, temporal_stride: int) -> tuple[int, ...]:
    if anchor < 0:
        raise ValueError("anchor must be non-negative")
    if sequence_length < 1 or temporal_stride < 1:
        raise ValueError("sequence_length and temporal_stride must be positive")
    return tuple(max(0, anchor-offset*temporal_stride)
                 for offset in range(sequence_length-1, -1, -1))
