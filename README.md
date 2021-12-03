# A Tale of Two Gradients
Testing gradient-based optimization for manipulation

## Installation

To install, run the following commands:

```bash
git clone https://github.com/dawsonc/a_tale_of_two_gradients
cd autograd_manipulation
conda create -n autograd_manipulation_env python=3.9
conda activate autograd_manipulation_env
pip install -e .
pip install -r requirements.txt
```

If you want to run in Jupyter notebooks, you also need to install the kernel
```bash
python -m ipykernel install --user --name=autograd_manipulation_env
```
