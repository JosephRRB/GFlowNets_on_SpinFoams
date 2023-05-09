# Virtual Environment Setup

module load python/3
virtualenv --no-download tensorflow
source tensorflow/bin/activate
pip install --no-index tensorflow==2.8

# Parameters to Test

- Spins and spinfoam model(obviously)
- Learning Rate
- Main layer hidden nodes
  - Restrictions on first and last layer
  - Last > output layer
  - First ~ input layer
- Different activation functions (swish, tanh, relu)
- Exploration rate (0, .25, .5, .75, 1)
- Training fraction (0, .25, .5, .75, 1)
  - Single = 0
  - Star = .5
- Evaluation batch size = 10e4
- Batch size = 10e3
- Number iterations = 10e4
- Generate samples every m should be the same as MHMC