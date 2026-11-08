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
   You provide it as ``HF_TOKEN`` (see below).

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
under the name ``HF_TOKEN`` and enable notebook access for it. The
notebook downloads the model in the *Download the CellSAM model* cell and
tells you there if the license has not been accepted yet.

Local installation
==================

1. Clone the repository and create a virtual environment.

   .. code-block:: bash

      git clone https://github.com/ronalabrcns/DuckSeg.git
      cd DuckSeg
      python -m venv myenv

   Activate it with ``source myenv/bin/activate`` (Linux, macOS) or
   ``myenv\Scripts\activate`` (Windows).

2. Install the requirements.

   .. code-block:: bash

      python -m pip install -r requirements.txt

   ``requirements.txt`` is a full, pinned snapshot of a Linux/NVIDIA
   environment (it includes CUDA packages). On other systems, install
   PyTorch for your platform first and then the package itself, which
   brings the remaining dependencies:

   .. code-block:: bash

      pip install -e .

3. Install `CellSAM <https://github.com/vanvalenlab/cellSAM>`_. It is not
   on PyPI and is therefore not part of the package dependencies. DuckSeg
   is tested with this commit:

   .. code-block:: bash

      pip install git+https://github.com/vanvalenlab/cellSAM.git@362a8e521bcd7313d5cb280bcd4cfc12977c1e9f

4. Accept the CellSAM license (see `CellSAM model license`_) and make your
   Hugging Face token available before starting Jupyter, either once with
   ``hf auth login`` or as an environment variable:

   .. code-block:: bash

      export HF_TOKEN="<your token>"     # Linux, macOS
      set HF_TOKEN=<your token>          # Windows (cmd)

   The weights are downloaded on the first segmentation (or explicitly with
   :func:`DuckSeg.model_loader.load_cellsam_model`). After that, no token is
   needed on this computer.

5. Start Jupyter and open a notebook from ``notebooks/``.

   .. code-block:: bash

      python -m jupyter lab

The interactive tools (ROI editor, review grid) are Jupyter widgets. In
classic Jupyter run ``%matplotlib widget`` before the ROI editor; this
needs the ``ipympl`` package.

Building this documentation
===========================

.. code-block:: bash

   pip install -e ".[docs]"
   sphinx-build -W -b html docs docs/_build/html
