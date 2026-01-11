import scripts.label_hand_crops as label_hand_crops


def test_labeler_help_defaults_to_true() -> None:
    args = label_hand_crops.parse_args(["--crops-jsonl", "crops.jsonl"])
    assert args.show_help is True


def test_labeler_help_can_be_disabled() -> None:
    args = label_hand_crops.parse_args(["--crops-jsonl", "crops.jsonl", "--no-help"])
    assert args.show_help is False


def test_labeler_show_help_parses_false() -> None:
    args = label_hand_crops.parse_args(["--crops-jsonl", "crops.jsonl", "--show-help", "false"])
    assert args.show_help is False
