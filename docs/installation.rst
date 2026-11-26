============
Installation
============

DuckSeg needs **Python 3.10 or newer**. A CUDA-capable GPU is strongly
recommended: CellSAM runs on every frame of every video, which is slow on
a CPU. If you have no local GPU, use the Google Colab notebook instead.

.. contents::
   :local:
   :depth: 1

CellSAM model license
=====================

DuckSeg itself is MIT-licensed. The CellSAM model weights it uses are
third-party assets from the Van Valen lab, licensed under a modified
Apache 2.0 license for **non-commercial academic use only**. They are
downloaded automatically the first time DuckSeg segments a frame (and
then cached in ``~/.deepcell/models``), but you have to accept the license
once first:

1. Log in at `huggingface.co <https://huggingface.co>`_ (a free account is
   enough), open the model page
   `vanvalenlab/cellsam <https://huggingface.co/vanvalenlab/cellsam>`_
   and click *I agree to the license terms*.
2. Create a *Read* token at
   `huggingface.co/settings/tokens <https://huggingface.co/settings/tokens>`_.
   A *fine-grained* token also works if *Read access to contents of all
   public gated repos you can access* is switched on. You provide the
   token as ``HF_TOKEN`` (see below).

A DeepCell access token from `users.deepcell.org <https://users.deepcell.org/>`_,
set as ``DEEPCELL_ACCESS_TOKEN``, still works instead of the Hugging Face
token. See :mod:`DuckSeg.model_loader` for the details.

Google Colab (no installation)
==============================

Open the example notebook
`DuckSeg_Example.ipynb <https://colab.research.google.com/github/ronalabrcns/DuckSeg/blob/main/notebooks/DuckSeg_Example.ipynb>`_
and choose a GPU runtime (*Runtime → Change runtime type*). The first cells
clone the repository, install CellSAM and restart the kernel (wait for the
session to reconnect before continuing). By default the notebook downloads
an example dataset; set ``DATA_SOURCE = "drive"`` and ``DRIVE_DIR`` to
analyse your own videos from Google Drive.

If you would rather not edit code, open
`DuckSeg_Example_Form.ipynb <https://colab.research.google.com/github/ronalabrcns/DuckSeg/blob/main/notebooks/DuckSeg_Example_Form.ipynb>`_
instead: the same pipeline as Colab forms (text fields, checkboxes,
sliders) with the code hidden.

Accept the CellSAM license and create a Hugging Face token (see
`CellSAM model license`_). Store the token in the Colab *Secrets* panel
under the name ``HF_TOKEN`` and enable notebook access for it. The code
notebook downloads the model in the *Download the CellSAM model* cell. In
the form notebook, tick ``i_accept_the_cellsam_license`` in step 2
(*Accept the license and load the CellSAM model*) before running it. Both
notebooks tell you there if the token is missing or the license has not
been accepted yet.

.. figure:: /_static/screenshots/google_colab_secrets.png
   :alt: Colab Secrets panel with an HF_TOKEN entry and notebook access switched on
   :width: 420px
   :align: center

   The Colab *Secrets* panel (key icon on the left): name ``HF_TOKEN``, your token as the value, notebook access on.

Local installation
==================

1. Clone the repository and create an environment, with conda:

   .. code-block:: bash

      git clone https://github.com/ronalabrcns/DuckSeg.git
      cd DuckSeg
      conda create -n duckseg python=3.12
      conda activate duckseg

   or with a virtual environment: ``python -m venv myenv``, then activate it
   with ``source myenv/bin/activate`` (Linux, macOS) or
   ``myenv\Scripts\activate`` (Windows).

2. Install the requirements and DuckSeg.

   .. code-block:: bash

      python -m pip install -r requirements.txt
      python -m pip install -e .

   ``requirements.txt`` holds only what DuckSeg needs, pinned to the tested
   versions, including `CellSAM <https://github.com/vanvalenlab/cellSAM>`_
   at the tested commit (CellSAM is not on PyPI) and PyTorch. It needs
   Python 3.11 or newer. For an NVIDIA GPU on Windows, install the CUDA build
   of PyTorch from `pytorch.org <https://pytorch.org/get-started/locally/>`_
   first; the default Windows build is CPU-only.

   Without ``requirements.txt`` (for example on Python 3.10), install the
   package, which brings its dependencies in their latest versions, and
   CellSAM at the tested commit:

   .. code-block:: bash

      pip install -e .
      pip install git+https://github.com/vanvalenlab/cellSAM.git@362a8e521bcd7313d5cb280bcd4cfc12977c1e9f

3. Accept the CellSAM license (see `CellSAM model license`_) and make your
   Hugging Face token available before starting Jupyter, either once with
   ``hf auth login`` or as an environment variable:

   .. code-block:: bash

      export HF_TOKEN="<your token>"     # Linux, macOS
      set HF_TOKEN=<your token>          # Windows (cmd)
      $env:HF_TOKEN="<your token>"       # Windows (PowerShell)

   The weights are downloaded on the first segmentation (or explicitly with
   :func:`DuckSeg.model_loader.load_cellsam_model`). After that, no token is
   needed on this computer.

4. Start Jupyter and open a notebook from ``notebooks/``.

   .. code-block:: bash

      python -m jupyter lab

   For your own data, open ``notebooks/DuckSeg_Local.ipynb``: set
   ``DATA_DIR`` to a folder with your videos and their ``*_ROI.tif`` masks
   (one condition per folder, see :doc:`guides/data_preparation`) and run all
   cells. Plots and the CSV are saved to ``duckseg_output`` inside that folder.

CellSAM version and reproducibility
===================================

DuckSeg results depend on the CellSAM model that segments the nuclei. The
analyses in the DuckSeg paper used the ``cellsam_general`` weights v1.2 with
the CellSAM commit pinned above. A newer CellSAM release, or new model
weights, can outline nuclei slightly differently, so the set of tracked and
retained cells, and therefore the *n* per condition and the exact curves,
may change even with identical DuckSeg settings. The qualitative
recruitment result should not depend on this, but to reproduce numbers
exactly:

- install CellSAM from the pinned commit,
- note the DuckSeg version or Git commit you used, and
- keep the transforms, ``range_modifier``, filter thresholds and manual
  exclusions together with your exported CSV files.

Fixes in DuckSeg itself can also change *n*: for example, since November
2026 the scored band is clamped to the nucleus crop, so cells that were
previously dropped when the widened band reached above the nucleus are now
scored.

The interactive tools (ROI editor, review grid) are Jupyter widgets. In
classic Jupyter run ``%matplotlib widget`` before the ROI editor; this
needs the ``ipympl`` package.

