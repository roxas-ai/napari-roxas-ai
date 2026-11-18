from unittest.mock import patch
from pathlib import Path
import yaml
import napari_roxas_ai
from napari_roxas_ai._docs import DOCS_URL, open_user_documentation


def test_open_user_documentation_calls_browser():
    with patch("webbrowser.open") as mock_open, patch("napari_roxas_ai._docs.show_info") as mock_info:
        open_user_documentation()
        mock_open.assert_called_once_with(DOCS_URL)
        mock_info.assert_called_once()
        assert "documentation" in mock_info.call_args[0][0].lower()


def test_docs_lazy_import():
    func = getattr(napari_roxas_ai, "open_user_documentation")
    assert callable(func)
    assert func is open_user_documentation


def test_manifest_documentation_contributions():
    manifest_path = Path(__file__).resolve().parent.parent / "napari.yaml"
    assert manifest_path.exists(), "napari.yaml not found"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)

    commands = manifest.get("contributions", {}).get("commands", [])
    doc_command = next(
        (cmd for cmd in commands if cmd.get("id") == "napari-roxas-ai.open_user_documentation"),
        None,
    )
    assert doc_command is not None, "Documentation command contribution not found in napari.yaml"
    assert doc_command.get("python_name") == "napari_roxas_ai:open_user_documentation"

    widgets = manifest.get("contributions", {}).get("widgets", [])
    doc_widget = next(
        (w for w in widgets if w.get("command") == "napari-roxas-ai.open_user_documentation"),
        None,
    )
    assert doc_widget is not None, "Documentation widget contribution not found in napari.yaml"
    assert doc_widget.get("display_name") == "── USER DOCUMENTATION (online) ──"
