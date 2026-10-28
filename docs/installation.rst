============
Installation
============

DuckSeg needs **Python 3.10 or newer**. A CUDA-capable GPU is strongly
recommended: CellSAM runs on every frame of every video, which is slow on
a CPU. If you have no local GPU, use the Google Colab notebook instead.

.. contents::
   :local:
   :depth: 1

Google Colab (no installation)
==============================

Open the example notebook
`DuckSeg_Example.ipynb <https://colab.research.google.com/github/ronalabrcns/DuckSeg/blob/main/notebooks/DuckSeg_Example.ipynb>`_
and choose a GPU runtime (*Runtime → Change runtime type*). The first cells
clone the repository, install CellSAM and restart the kernel (wait for the
session to reconnect before continuing). By default the notebook downloads
an example dataset; set ``DATA_SOURCE = "drive"`` and ``DRIVE_DIR`` to
analyse your own videos from Google Drive.

CellSAM needs a DeepCell access token. Create a token on
`users.deepcell.org <https://users.deepcell.org/>`_, store it in the Colab
*Secrets* panel under the name ``DEEPCELL_ACCESS_TOKEN`` and enable
notebook access for it. The notebook copies it into the environment
variable that CellSAM reads.

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
   on PyPI and is therefore not part of the package dependencies.

   .. code-block:: bash

      pip install git+https://github.com/vanvalenlab/cellSAM.git

4. Make your DeepCell access token (see above) available before starting
   Jupyter.

   .. code-block:: bash

      export DEEPCELL_ACCESS_TOKEN="<your token>"     # Linux, macOS
      set DEEPCELL_ACCESS_TOKEN=<your token>           # Windows (cmd)

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
