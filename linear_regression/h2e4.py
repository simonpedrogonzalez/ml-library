import sys, os; sys.path.insert(0, os.path.abspath('.')) if os.path.abspath('.') not in sys.path else None
from linear_regression.gradient_descent import StochasticGradientDescent, BatchGradientDescent