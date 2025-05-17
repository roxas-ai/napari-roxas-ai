from pathlib import Path

from napari_roxas_ai._assets_files import check_assets_and_download

from ._data_loader import load_sample_data

__all__ = ("load_sample_data",)


BASE_DIR = Path(__file__).parent.absolute()
check_assets_and_download(str(BASE_DIR / "sample_data"), "sample_data.zip")
