import numpy as np
from sympy import symbols, sympify
from sympy.utilities.lambdify import lambdify
from gplearn.genetic import SymbolicRegressor
import json
import math

def main():
    with open('testdata.txt', 'r') as file:
        dat = file.read()
        data = json.loads(dat)
        # NumPy array 
        data = np.array(data)

    X = data[:, :-1]  # Features
    y = data[:, -1]   # Output

    #symbolic regression model
    function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'log']
    est_gp = SymbolicRegressor(population_size=1000,
                               generations=1000, stopping_criteria=0.01,
                               function_set=function_set,
                               p_crossover=0.7,
                               p_subtree_mutation=0.1,
                               p_hoist_mutation=0.05,
                               p_point_mutation=0.1,
                               max_samples=0.9, verbose=1,
                               parsimony_coefficient=0.01, random_state=0)

    est_gp.fit(X, y)

    converter = {
        'sub': lambda x, y : x - y,
        'div': lambda x, y : x/y,
        'mul': lambda x, y : x*y,
        'add': lambda x, y : x + y,
        'neg': lambda x    : -x,
        'pow': lambda x, y : x**y,
        'sin': lambda x    : sin(x),
        'cos': lambda x    : cos(x),
        'inv': lambda x: 1/x,
        'sqrt': lambda x: x**0.5,
        'pow3': lambda x: x**3
    }

    sympy_expression = sympify(str(est_gp._program), locals=converter)
    print("Best symbolic expression found:")
    print(sympy_expression)

    predict_func = lambdify([symbols('x' + str(i)) for i in range(X.shape[1])], sympy_expression)

    # Sample output predictions
    #print("Sample output predictions:")
    #for i in range(X.shape[0]):
    #    inputs = X[i]
    #    prediction = predict_func(*inputs)
    #    print("Predicted:", prediction, "Actual:", y[i])

def sub(x,y):
    return (x - y),



main()
