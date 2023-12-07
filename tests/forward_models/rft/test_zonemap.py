import argparse

import pytest

from semeio.forward_models.rft.zonemap import ZoneMap


def test_zonemap():
    zones_at_k_value = {1: ["Zone1", "Zone2"], 3: ["Zone3"]}
    zone_map = ZoneMap(zones_at_k_value)
    assert zone_map.has_relationship("Zone1", 1)
    assert zone_map.has_relationship("Zone2", 1)
    assert not zone_map.has_relationship("Zone1", 3)
    assert zone_map.has_relationship("Zone3", 3)


@pytest.fixture()
def initdir(tmpdir):
    tmpdir.chdir()
    valid_data = """
1 zone1 zone3
2 zone2
3 zone2
"""
    tmpdir.join("valid_zonemap.txt").write(valid_data)
    comment = "-- this is a comment"
    valid_data = comment + "\n" + valid_data + comment
    tmpdir.join("valid_zonemap_with_comments.txt").write(valid_data)

    valid_weird_data = """
1 zone1 zone3 2 3
2 zone2
3 zone2
"""
    tmpdir.join("valid_weird_zonemap.txt").write(valid_weird_data)

    invalid_data = """
0 zone1
2 zone2
3 zone2
"""
    tmpdir.join("invalid_starts_at_0.txt").write(invalid_data)

    invalid_data = """
zone1 2
2 zone2
3 zone2
"""
    tmpdir.join("invalid_format_string_first.txt").write(invalid_data)

    invalid_data = """
1
2 zone2
3 zone2
"""
    tmpdir.join("invalid_format_missing_zone.txt").write(invalid_data)


@pytest.mark.usefixtures("initdir")
def test_load_data():
    for fname in [
        "valid_zonemap.txt",
        "valid_zonemap_with_comments.txt",
        "valid_weird_zonemap.txt",
    ]:
        zone_map = ZoneMap.load_and_parse_zonemap_file(fname)
        assert zone_map.has_relationship("zone1", 0)
        assert zone_map.has_relationship("zone3", 0)
        assert zone_map.has_relationship("zone2", 2)
        assert zone_map.has_relationship("zone2", 1)
        assert not zone_map.has_relationship("zone1", 1)
        assert not zone_map.has_relationship("zone1", 2)


@pytest.mark.usefixtures("initdir")
def test_invalid_load():
    errors = [
        "k values cannot be 0, must start at 1",
        "k must be integer, was zone1",
        "Number of zonenames must be 1 or more",
        "ZoneMap file non_existing not found",
    ]

    fnames = [
        "invalid_starts_at_0.txt",
        "invalid_format_string_first.txt",
        "invalid_format_missing_zone.txt",
        "non_existing",
    ]

    for fname, error in zip(fnames, errors):
        with pytest.raises(argparse.ArgumentTypeError) as msgcontext:
            ZoneMap.load_and_parse_zonemap_file(fname)
        assert error in msgcontext.value.args[0]
