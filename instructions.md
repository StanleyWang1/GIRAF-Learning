# Environment setup

1. Install [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) if `conda --version` is not available.

2. From this repository, create and activate a Python 3.12 environment:

   ```bash
   conda create --name giraf-learning python=3.12 pip
   conda activate giraf-learning
   ```

3. Install the project dependencies into the active environment:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

4. Verify the installation:

   ```bash
   python -m pip check
   python -c "import pyCandle, dynamixel_sdk, numpy, matplotlib, sympy; print('Dependencies OK')"
   ```

After changing `requirements.txt`, reactivate the environment and rerun `python -m pip install -r requirements.txt`.
