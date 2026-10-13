from __future__ import annotations

Offsets = list[tuple[int, int]]


def exact_end_pairs(sender_offsets: Offsets,
                    receiver_offsets: Offsets) -> list[tuple[int, int]]:
    receiver_by_end: dict[int, int] = {}
    for index, (start, end) in enumerate(receiver_offsets):
        if end > start:
            receiver_by_end.setdefault(end, index)
    return [(index, receiver_by_end[end])
            for index, (start, end) in enumerate(sender_offsets)
            if end > start and end in receiver_by_end]


def receiver_to_sender_indices(sender_offsets: Offsets,
                               receiver_offsets: Offsets) -> list[int]:
    matched = {receiver: sender for sender, receiver
               in exact_end_pairs(sender_offsets, receiver_offsets)}
    previous = None
    result = []
    for index in range(len(receiver_offsets)):
        if index in matched:
            previous = matched[index]
        result.append(0 if previous is None else previous)
    return result


def proportional_layers(sender_layers: int, receiver_layers: int) -> list[int]:
    return [round(layer * (sender_layers - 1) / (receiver_layers - 1))
            for layer in range(receiver_layers)]


def stencil_taps(sender_layers: int, receiver_layers: int, span: int = 1,
                 stride: int = 4) -> list[list[int]]:
    centers = proportional_layers(sender_layers, receiver_layers)
    return [[min(max(center + offset * stride, 0), sender_layers - 1)
             for offset in range(-span, span + 1)]
            for center in centers]
