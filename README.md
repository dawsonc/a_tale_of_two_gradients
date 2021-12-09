# A Tale of Two Gradients
Exploring gradient-based optimization for manipulation. For more information, see our video [here](https://youtu.be/GpS5OB7l-dY) and report in `report.pdf`.

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

## Running

All experiments can be run from the `experiments` directory, and will save their progress in `experiments/results`.

- `python experiments/first_order_optimization.py` will solve both the box flipping and box grasping tasks using gradient descent with exact gradients.
- `python experiments/zero_order_optimization.py` will solve both the box flipping and box grasping tasks using gradient descent with stochastically approximated gradients. 
- `python experiments/cost_landscape.py` will save plots of the cost landscape for both problems.
- `python experiments/plot_results.py` will generate the plots and animations used in our report and video.
