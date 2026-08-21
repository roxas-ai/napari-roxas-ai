"""
Tests for upgrading a settings.json written by an older version.

The settings file is not part of the repository: it lives next to the
installed package and survives a `pip uninstall`, so any version of the file
can show up under a current install. These tests pin down the contract that
makes this safe: user values are never lost, missing settings are filled in
from the defaults, and keys that were renamed in the past keep their value
under the current name.
"""

import json
from copy import deepcopy

import pytest

from napari_roxas_ai._settings._settings_manager import (
    DEFAULT_SETTINGS,
    SettingsManager,
    upgrade_settings,
)
from napari_roxas_ai._utils._metadata_keys import migrate_legacy_metadata_keys


def _field_ids(settings):
    return [field["id"] for field in settings["samples_metadata"]["fields"]]


def _legacy_settings():
    """A settings.json as written before the metadata key renames."""
    return {
        "samples_metadata": {
            "fields": [
                {
                    "id": "sample_name",
                    "label": "Name",
                    "widget_type": "QLineEdit",
                    "read_only": True,
                    "required": True,
                },
                {
                    "id": "sample_type",
                    "label": "Type",
                    "widget_type": "QComboBox",
                    "items": ["conifer", "angiosperm"],
                    "editable": True,
                    "required": True,
                },
                {
                    "id": "sample_geometry",
                    "label": "Geometry",
                    "widget_type": "QComboBox",
                    "items": ["linear", "circular"],
                    "editable": True,
                    "required": True,
                },
                {
                    "id": "sample_scale",
                    "label": "Scale (px/µm)",
                    "widget_type": "QDoubleSpinBox",
                    "default": 4.0,
                    "min": 0.001,
                    "max": 1000.0,
                    "step": 0.01,
                    "decimals": 4,
                    "required": True,
                },
            ]
        },
        "processing": {
            "try_to_use_gpu": True,
            "try_to_use_autocast": True,
        },
        "measurements": {
            "cells_cluster_separation_threshold": 2.5,
            "cells_smoothing_kernel_size": 5,
            "cells_integration_interval": 0.5,
            "cells_tangential_angle": 0.0,
        },
        "project_directory": "/some/where/MyProject",
    }


def test_user_values_are_kept():
    """Settings the user changed must survive the upgrade."""
    upgraded = upgrade_settings(_legacy_settings())

    assert upgraded["processing"]["try_to_use_gpu"] is True
    assert upgraded["processing"]["try_to_use_autocast"] is True
    assert upgraded["project_directory"] == "/some/where/MyProject"
    assert upgraded["measurements"]["cells_smoothing_kernel_size"] == 5


def test_missing_settings_are_filled_in():
    """Settings added by a newer version must appear with their default."""
    upgraded = upgrade_settings(_legacy_settings())

    # Whole blocks that the legacy file did not have at all
    assert upgraded["file_extensions"] == DEFAULT_SETTINGS["file_extensions"]
    assert upgraded["rasterization"] == DEFAULT_SETTINGS["rasterization"]

    # Single keys added to an existing block
    measurements = upgraded["measurements"]
    for key in (
        "lower_limit_cwt_iqr_multiplier",
        "upper_limit_cwt_iqr_multiplier",
        "opposite_cwt_ratio_limit",
        "adjacent_cwt_ratio_limit",
    ):
        assert measurements[key] == DEFAULT_SETTINGS["measurements"][key]

    # A metadata field added to an existing field list
    assert "rings_outmost_complete_year" in _field_ids(upgraded)


def test_renamed_measurement_keys_keep_their_value():
    upgraded = upgrade_settings(_legacy_settings())
    measurements = upgraded["measurements"]

    assert measurements["cluster_dbl_cwt_threshold"] == 2.5
    assert measurements["relwidth_cwt_integration"] == 0.5
    assert "cells_cluster_separation_threshold" not in measurements
    assert "cells_integration_interval" not in measurements


def test_renamed_metadata_fields_are_migrated():
    """
    The field ids end up verbatim as keys of a sample's .metadata.json, so a
    stale id makes freshly prepared samples unreadable by the current code.
    """
    upgraded = upgrade_settings(_legacy_settings())
    ids = _field_ids(upgraded)

    assert "meas_geometry" in ids
    assert "spatial_resolution" in ids
    assert "sample_geometry" not in ids
    assert "sample_scale" not in ids

    # The renamed field takes the current definition, but keeps the value the
    # user had configured
    resolution = next(
        field
        for field in upgraded["samples_metadata"]["fields"]
        if field["id"] == "spatial_resolution"
    )
    assert resolution["default"] == 4.0
    assert resolution["label"] == "Spatial Resolution (px/µm)"


def test_field_order_follows_the_defaults():
    upgraded = upgrade_settings(_legacy_settings())
    assert _field_ids(upgraded) == _field_ids(DEFAULT_SETTINGS)


def test_user_added_field_is_kept():
    legacy = _legacy_settings()
    custom = {"id": "my_own_field", "label": "Mine", "widget_type": "QLineEdit"}
    legacy["samples_metadata"]["fields"].append(custom)

    upgraded = upgrade_settings(legacy)
    assert custom in upgraded["samples_metadata"]["fields"]


def test_upgrading_current_settings_is_a_no_op():
    assert upgrade_settings(DEFAULT_SETTINGS) == DEFAULT_SETTINGS


def test_upgrade_is_idempotent():
    once = upgrade_settings(_legacy_settings())
    assert upgrade_settings(once) == once


def test_defaults_are_not_mutated():
    before = json.dumps(DEFAULT_SETTINGS, sort_keys=True)
    upgrade_settings(_legacy_settings())["measurements"][
        "cluster_dbl_cwt_threshold"
    ] = 999
    assert json.dumps(DEFAULT_SETTINGS, sort_keys=True) == before


def test_legacy_file_on_disk_is_upgraded_and_rewritten(tmp_path, monkeypatch):
    """A stale settings.json left behind by an older install is repaired."""
    settings_file = tmp_path / "settings.json"
    settings_file.write_text(json.dumps(_legacy_settings()), encoding="utf-8")

    monkeypatch.setattr(SettingsManager, "_instance", None)
    monkeypatch.setattr(SettingsManager, "_settings_file", settings_file)

    manager = SettingsManager()

    assert manager.get("processing.try_to_use_gpu") is True
    assert manager.get("measurements.cluster_dbl_cwt_threshold") == 2.5

    # The repaired settings are written back to disk
    on_disk = json.loads(settings_file.read_text(encoding="utf-8"))
    assert _field_ids(on_disk) == _field_ids(DEFAULT_SETTINGS)


def test_legacy_sample_metadata_keys_are_migrated():
    """Samples prepared with an older version must stay readable."""
    legacy_metadata = {
        "sample_name": "NF_522A_1_2",
        "sample_type": "conifer",
        "sample_geometry": "linear",
        "sample_scale": 2.2675,
        "rings_outmost_complete_year": 1983,
        "scan_format": "JPEG",
    }

    migrated = migrate_legacy_metadata_keys(legacy_metadata)

    assert migrated["meas_geometry"] == "linear"
    assert migrated["spatial_resolution"] == 2.2675
    assert "sample_geometry" not in migrated
    assert "sample_scale" not in migrated

    # The renamed keys keep their position, so a migrated file looks like a
    # freshly written one
    assert list(migrated) == [
        "sample_name",
        "sample_type",
        "meas_geometry",
        "spatial_resolution",
        "rings_outmost_complete_year",
        "scan_format",
    ]


def test_current_sample_metadata_wins_over_legacy_key():
    metadata = {
        "sample_scale": 1.0,
        "spatial_resolution": 2.2675,
    }
    assert migrate_legacy_metadata_keys(metadata) == {
        "spatial_resolution": 2.2675
    }


def test_current_sample_metadata_is_untouched():
    metadata = {"sample_name": "x", "spatial_resolution": 2.2675}
    assert migrate_legacy_metadata_keys(metadata) == metadata


def _edited_settings():
    """A current settings.json with the kind of edits a user makes by hand."""
    edited = deepcopy(DEFAULT_SETTINGS)
    edited["processing"]["try_to_use_gpu"] = True
    edited["measurements"]["cells_tangential_angle"] = 42.0
    edited["my_own_block"] = {"hello": "world"}
    edited["rasterization"]["rings_color_sequence"] = ["red", "blue"]
    del edited["tables"]["separator"]  # a setting a newer version added
    return edited


def _load_file(path, monkeypatch):
    monkeypatch.setattr(SettingsManager, "_instance", None)
    monkeypatch.setattr(SettingsManager, "_settings_file", path)
    return SettingsManager()


def _assert_edits_survived(manager, custom=None):
    """The four rules a hand-edited settings.json must obey on load."""
    custom = {"hello": "world"} if custom is None else custom
    # Key present: the user value is not overwritten
    assert manager.get("processing.try_to_use_gpu") is True
    assert manager.get("measurements.cells_tangential_angle") == 42.0
    # Key only in the user file: left alone
    assert manager.get("my_own_block") == custom
    # List: the user list is not overwritten
    assert manager.get("rasterization.rings_color_sequence") == ["red", "blue"]
    # Key missing: filled in from the defaults
    assert manager.get("tables.separator") == ";"


@pytest.mark.parametrize(
    "encoding", ["utf-8", "utf-8-sig"], ids=["utf8", "utf8_with_bom"]
)
def test_hand_edited_file_loads_with_or_without_a_bom(
    encoding, tmp_path, monkeypatch
):
    """
    The file is meant to be hand-edited, and Windows editors add a UTF-8 byte
    order mark when saving. Reading with plain "utf-8" rejected such a file,
    which used to cost the user every value they had set.
    """
    edited = _edited_settings()
    edited["my_own_block"] = {"note": "Grösse in µm"}  # non-ASCII on purpose

    settings_file = tmp_path / "settings.json"
    settings_file.write_bytes(
        json.dumps(edited, indent=4, ensure_ascii=False).encode(encoding)
    )

    manager = _load_file(settings_file, monkeypatch)

    assert manager.get("my_own_block") == {"note": "Grösse in µm"}
    _assert_edits_survived(manager, custom={"note": "Grösse in µm"})


def test_edits_are_kept_when_defaults_are_filled_in(tmp_path, monkeypatch):
    """Filling in a missing setting must not touch the surrounding values."""
    edited = _edited_settings()
    del edited["measurements"]["adjacent_cwt_ratio_limit"]

    settings_file = tmp_path / "settings.json"
    settings_file.write_text(json.dumps(edited, indent=4), encoding="utf-8")

    manager = _load_file(settings_file, monkeypatch)

    assert manager.get("measurements.adjacent_cwt_ratio_limit") == 3.0
    _assert_edits_survived(manager)

    # ...and the completed settings are on disk, still carrying the edits
    on_disk = json.loads(settings_file.read_text(encoding="utf-8"))
    assert on_disk["my_own_block"] == {"hello": "world"}
    assert on_disk["measurements"]["adjacent_cwt_ratio_limit"] == 3.0


def test_missing_file_is_created_from_the_defaults(tmp_path, monkeypatch):
    settings_file = tmp_path / "settings.json"

    manager = _load_file(settings_file, monkeypatch)

    assert json.loads(settings_file.read_text(encoding="utf-8")) == json.loads(
        json.dumps(DEFAULT_SETTINGS)
    )
    assert manager.get("processing.try_to_use_gpu") is False


@pytest.mark.parametrize(
    "content",
    ['{"processing": {"try_to_use_gpu": true,,}}', "[1, 2, 3]", "not json"],
    ids=["stray_comma", "not_an_object", "garbage"],
)
def test_unreadable_file_is_left_untouched(
    content, tmp_path, monkeypatch, recwarn
):
    """
    A typo in a hand-edited file must not cost the user their settings: the
    file stays exactly as it is so it can be corrected, and the reason is
    reported instead of failing silently.
    """
    settings_file = tmp_path / "settings.json"
    settings_file.write_text(content, encoding="utf-8")

    manager = _load_file(settings_file, monkeypatch)

    assert settings_file.read_text(encoding="utf-8") == content
    assert manager.get("processing.try_to_use_gpu") is False
    assert any("settings.json" in str(w.message) for w in recwarn)


def test_reset_is_the_only_thing_that_discards_user_values(
    tmp_path, monkeypatch
):
    """reset() is what a "reset to defaults" button calls, and only that."""
    settings_file = tmp_path / "settings.json"
    settings_file.write_text(
        json.dumps(_edited_settings(), indent=4), encoding="utf-8"
    )

    manager = _load_file(settings_file, monkeypatch)
    _assert_edits_survived(manager)

    manager.reset()

    assert manager.get("processing.try_to_use_gpu") is False
    assert manager.get("my_own_block") is None
    assert (
        manager.get("rasterization.rings_color_sequence")
        == DEFAULT_SETTINGS["rasterization"]["rings_color_sequence"]
    )
