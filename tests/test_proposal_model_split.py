from __future__ import annotations

from src.proposal_model import StateRoleSample, split_state_role_samples


def _make_samples(with_stem: bool = True) -> list[StateRoleSample]:
    samples: list[StateRoleSample] = []
    stems = ["batch1", "batch2", "batch3"]
    for stem in stems:
        for idx in range(2):
            samples.append(
                StateRoleSample(
                    in_hand=[1.0 if i == idx else 0.0 for i in range(8)],
                    role=(idx % 8) + 1,
                    stem=stem if with_stem else None,
                )
            )
    return samples


def test_group_split_keeps_stems_together() -> None:
    samples = _make_samples(with_stem=True)
    train_samples, val_samples = split_state_role_samples(
        samples, val_split=0.4, seed=7, split_mode="group"
    )
    train_stems = {sample.stem for sample in train_samples}
    val_stems = {sample.stem for sample in val_samples}
    assert train_stems.isdisjoint(val_stems)
    stem_to_split = {}
    for sample in samples:
        split = "train" if sample in train_samples else "val"
        stem_to_split.setdefault(sample.stem, split)
        assert stem_to_split[sample.stem] == split


def test_group_split_reproducible_with_seed() -> None:
    samples = _make_samples(with_stem=True)
    _, val_a = split_state_role_samples(samples, val_split=0.4, seed=123, split_mode="group")
    _, val_b = split_state_role_samples(samples, val_split=0.4, seed=123, split_mode="group")
    val_ids_a = {id(sample) for sample in val_a}
    val_ids_b = {id(sample) for sample in val_b}
    assert val_ids_a == val_ids_b


def test_row_split_works_without_stems() -> None:
    samples = _make_samples(with_stem=False)
    train_samples, val_samples = split_state_role_samples(samples, val_split=0.4, seed=5)
    assert len(train_samples) + len(val_samples) == len(samples)
