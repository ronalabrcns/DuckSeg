"""
Download and load the CellSAM model weights.

The CellSAM weights are third-party assets, licensed by the Van Valen lab
under a modified Apache 2.0 license for **non-commercial academic use
only** (separate from DuckSeg's MIT license). They can be obtained in two
ways:

- **Hugging Face** (recommended): the gated repository
  `vanvalenlab/cellsam <https://huggingface.co/vanvalenlab/cellsam>`_. Log
  in, accept the license on the model page once, and provide a read token
  as ``HF_TOKEN`` (or log in with ``hf auth login``).
- **DeepCell**: a ``DEEPCELL_ACCESS_TOKEN`` from
  `users.deepcell.org <https://users.deepcell.org/>`_, which CellSAM's own
  :func:`cellSAM.get_model` uses.

Both sources deliver the same archive, which is unpacked into
``~/.deepcell/models/cellsam_v<version>/``, where :func:`cellSAM.get_model`
finds it. Once the weights are there, no token is needed any more.

:func:`load_cellsam_model` is what :func:`DuckSeg.experiment_evaluator.generate_mask`
calls on first use.
"""

__all__ = [
    "HF_REPO_ID",
    "HF_MODEL_URL",
    "MODEL_VERSION",
    "weights_available",
    "download_weights_from_hf",
    "load_cellsam_model",
]

import os
import shutil
import tarfile
import tempfile
from pathlib import Path

HF_REPO_ID = "vanvalenlab/cellsam"
HF_MODEL_URL = f"https://huggingface.co/{HF_REPO_ID}"
MODEL_VERSION = "1.2"

_MODELS_DIR = Path.home() / ".deepcell" / "models"

_LICENSE_HELP = (
    "The CellSAM weights are licensed for non-commercial academic use only.\n"
    "To download them from Hugging Face:\n"
    f"  1) Log in at huggingface.co, open {HF_MODEL_URL} and click\n"
    "     'I agree to the license terms'.\n"
    "  2) Create a read token at https://huggingface.co/settings/tokens\n"
    "  3) Set it as the HF_TOKEN environment variable (in Colab: Secrets panel,\n"
    "     with notebook access on), or run `hf auth login`.\n"
    "Alternatively set DEEPCELL_ACCESS_TOKEN (from https://users.deepcell.org)."
)


def _model_path(version: str, model: str) -> Path:
    return _MODELS_DIR / f"cellsam_v{version}" / f"{model}.pt"


def weights_available(version: str = MODEL_VERSION, model: str = "cellsam_general") -> bool:
    """
    Return whether the CellSAM weights are already downloaded.

    Parameters
    ----------
    version : str, default "1.2"
        CellSAM model version.
    model : str, default "cellsam_general"
        Model name inside the archive (``"cellsam_general"`` or ``"cellsam_extra"``).

    Returns
    -------
    bool
        ``True`` if ``~/.deepcell/models/cellsam_v<version>/<model>.pt`` exists.
    """
    return _model_path(version, model).is_file()


def _hf_token(token=None):
    if token:
        return token
    try:
        from huggingface_hub import get_token
        return get_token()          # HF_TOKEN or the token saved by `hf auth login`
    except ImportError:
        return os.environ.get("HF_TOKEN")


def download_weights_from_hf(token: str = None, version: str = MODEL_VERSION,
                             model: str = "cellsam_general") -> Path:
    """
    Download the CellSAM weights from Hugging Face and unpack them.

    The license must have been accepted on :data:`HF_MODEL_URL` with the
    account the token belongs to.

    Parameters
    ----------
    token : str, optional
        Hugging Face read token. Defaults to ``HF_TOKEN`` or the token saved
        by ``hf auth login``.
    version : str, default "1.2"
        CellSAM model version.
    model : str, default "cellsam_general"
        Model that must be present after unpacking.

    Returns
    -------
    pathlib.Path
        Path of the unpacked ``<model>.pt`` file.

    Raises
    ------
    RuntimeError
        If the license was not accepted, the token is missing/invalid, or the
        archive does not contain the expected model.
    """
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import GatedRepoError, HfHubHTTPError
    except ImportError as e:
        raise RuntimeError("huggingface_hub is not installed: pip install huggingface_hub") from e

    try:
        archive = hf_hub_download(HF_REPO_ID, f"cellsam-models_v{version}.tar.gz",
                                  token=_hf_token(token))
    except GatedRepoError as e:
        raise RuntimeError("Access to the CellSAM weights was refused: the license has not "
                           "been accepted for this Hugging Face account, or no token was "
                           "given.\n\n" + _LICENSE_HELP) from e
    except HfHubHTTPError as e:
        raise RuntimeError(f"Could not download the CellSAM weights from Hugging Face ({e}).\n\n"
                           + _LICENSE_HELP) from e

    # Unpack into a temporary folder first, then move the folder holding
    # <model>.pt to ~/.deepcell/models/cellsam_v<version>, where
    # cellSAM.get_model looks for it.
    target = _model_path(version, model).parent
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=_MODELS_DIR) as tmp:
        with tarfile.open(archive) as tar:
            if hasattr(tarfile, "data_filter"):
                tar.extractall(tmp, filter="data")
            else:
                tar.extractall(tmp)
        found = next(Path(tmp).rglob(f"{model}.pt"), None)
        if found is None:
            raise RuntimeError(f"{model}.pt not found in the downloaded archive {archive}.")
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        for f in found.parent.iterdir():
            shutil.move(str(f), str(target / f.name))
    return _model_path(version, model)


def load_cellsam_model(source: str = "auto", token: str = None,
                       version: str = MODEL_VERSION, model: str = "cellsam_general"):
    """
    Load the CellSAM model, downloading the weights first if needed.

    Parameters
    ----------
    source : {"auto", "huggingface", "deepcell"}, default "auto"
        Where to download the weights from if they are not on disk yet.
        ``"auto"`` uses Hugging Face when a Hugging Face token is available,
        otherwise DeepCell when ``DEEPCELL_ACCESS_TOKEN`` is set.
    token : str, optional
        Hugging Face read token (see :func:`download_weights_from_hf`).
    version : str, default "1.2"
        CellSAM model version.
    model : str, default "cellsam_general"
        ``"cellsam_general"`` or ``"cellsam_extra"``.

    Returns
    -------
    torch.nn.Module
        The loaded CellSAM model (on the CPU).

    Raises
    ------
    RuntimeError
        If the weights are not on disk and no download source is configured,
        or the download fails.
    ValueError
        If ``source`` is not one of the accepted values.
    """
    from cellSAM import get_model

    if source not in ("auto", "huggingface", "deepcell"):
        raise ValueError(f"source must be 'auto', 'huggingface' or 'deepcell', not {source!r}")

    if not weights_available(version, model):
        use_hf = source == "huggingface" or (source == "auto" and _hf_token(token))
        if use_hf:
            print(f"Downloading CellSAM weights from {HF_MODEL_URL} ...")
            download_weights_from_hf(token, version, model)
        elif source == "auto" and not os.environ.get("DEEPCELL_ACCESS_TOKEN"):
            raise RuntimeError("No CellSAM weights found and no token to download them.\n\n"
                               + _LICENSE_HELP)
        # else: cellSAM.get_model downloads them with DEEPCELL_ACCESS_TOKEN

    return get_model(model=model, version=version)
