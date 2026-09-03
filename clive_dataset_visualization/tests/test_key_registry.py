from dataset_visualization.backend.key_registry import build_key_groups, parse_camera_key


def test_parse_camera_key_rgb():
    parsed = parse_camera_key("camera_230322274541_rgb")
    assert parsed == ("230322274541", "rgb")


def test_parse_camera_key_cropped():
    parsed = parse_camera_key("camera_230322274541_rgb_cropped")
    assert parsed == ("230322274541_cropped", "rgb")


def test_build_key_groups_contains_expected_keys():
    keys = ["action", "joint_pos_L", "L_joint_pos_L", "joints_tau_external_L"]
    groups = build_key_groups(keys, keys)
    assert "action" in groups["Core"]
    assert "L_joint_pos_L" in groups["Leader vs Follower"]
    assert "joints_tau_external_L" in groups["Forces"]
