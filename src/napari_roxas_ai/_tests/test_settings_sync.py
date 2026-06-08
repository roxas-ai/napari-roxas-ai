import json
from pathlib import Path
from napari_roxas_ai._settings import SettingsManager

def test_settings_sync():
    """
    Test that the settings.json file in the repository (the 'template')
    is in sync with the hard-coded defaults in SettingsManager.
    
    This ensures that developers don't forget to update the JSON file
    when they add new default settings to the Python code.
    """
    # 1. Instantiate SettingsManager and get its defaults
    # Since SettingsManager is a singleton and _load_settings calls _set_defaults,
    # we can't easily get *only* the defaults without affecting the singleton state.
    # However, for a fresh test run, it should be fine.
    
    manager = SettingsManager()
    
    # We want to check if all default keys exist in the repository's settings.json
    # Note: SettingsManager.settings_file points to the actual file on disk.
    repo_settings_path = manager.settings_file
    
    assert repo_settings_path.exists(), f"Settings file not found at {repo_settings_path}"
    
    with open(repo_settings_path, 'r') as f:
        repo_settings = json.load(f)
    
    # 2. Get the current defaults by resetting the manager to defaults (temporarily)
    # We can't easily do this without side effects, but let's assume we can 
    # inspect the internal _settings after a reset.
    # We'll use a copy to compare.
    
    manager.reset()
    current_defaults = manager._settings.copy()
    
    # Remove the "_comment" if it exists in repo_settings but not in defaults
    if "_comment" in repo_settings and "_comment" not in current_defaults:
        del repo_settings["_comment"]

    # 3. Compare keys and structure recursively
    def compare_dicts(defaults, repo, path=""):
        for key, val in defaults.items():
            current_path = f"{path}.{key}" if path else key
            assert key in repo, f"Missing key '{current_path}' in settings.json. Please update it from SettingsManager._set_defaults()."
            
            if isinstance(val, dict):
                assert isinstance(repo[key], dict), f"Key '{current_path}' should be a dictionary in settings.json."
                compare_dicts(val, repo[key], current_path)
            # For lists of dicts (like 'fields'), we check if the items have the same keys
            elif isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
                assert len(val) == len(repo[key]), f"List length mismatch for '{current_path}'."
                for i, item in enumerate(val):
                    compare_dicts(item, repo[key][i], f"{current_path}[{i}]")

    compare_dicts(current_defaults, repo_settings)
    
    print("Settings sync verification successful!")

if __name__ == "__main__":
    test_settings_sync()
