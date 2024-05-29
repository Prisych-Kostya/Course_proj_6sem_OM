import numpy as np
from Functions import Function
from Hooke_jeeves import hooke_jeeves, hooke_jeeves_modified, hooke_jeeves_modified_with_constraints
import graph

if __name__ == '__main__':

    eps = 0.0001
    max_iter = 1000

    def stepeneva_func(*x):
        return (10*((x[0] - x[1])**2) + (x[0] - 1)**2)**4
    
    fun = Function(stepeneva_func, starting_point=[-1.2, 0.0], point_min=[1.0, 1.0])
    
    # 2.2.1
    points, iter, _ = hooke_jeeves(fun, step=0.4, alpha=0.18, epsilon=eps, long_criterion=False, max_iterations=max_iter, print_stats=True)
    graph.graph_search(points, fun)
    

    graph.graph_iterations_num_of_dx(fun, 0.1, 5, 0.05, alpha=0.5, epsilon=eps, max_iterations=max_iter)
    graph.graph_iterations_num_of_dx(fun, 0.1, 20, 0.1, alpha=0.5, epsilon=eps, max_iterations=max_iter)
    # 2.2.2
    optimal_step = 0.4
    graph.graph_iterations_num_of_alpha(fun, step=optimal_step, min_alpha=0.005, max_alpha=0.995, delta=0.005, epsilon=eps, max_iterations=max_iter)

    
    # 2.2.3
    steps_list = list(np.arange(0.1, 20, 0.2))
    graph.graph_long_short_criterion_fixed_alpha(fun, steps_list, alpha=0.18, epsilon=eps, max_iterations=max_iter)

    alpha_list = list(np.arange(0.005, 0.995, 0.005))
    graph.graph_long_short_criterion_fixed_step(fun, step=0.4, alphas=alpha_list, epsilon=eps, max_iterations=max_iter)


    # 2.2.4
    steps_list = list(np.arange(0.1, 2, 0.01))
    graph.graph_beta_difference_fixed_alpha(fun, steps_list, alpha=0.18, beta=1.0, epsilon=eps, max_iterations=max_iter)
    
    graph.graph_beta_difference_fixed_alpha(fun, steps_list, alpha=0.18, beta=1.2, epsilon=eps, max_iterations=max_iter)

    graph.graph_beta_difference_fixed_alpha_all(fun, steps_list, alpha=0.18, beta=1.2, epsilon=eps, max_iterations=max_iter)

    # Example of divergent search
    points, iter, _ = hooke_jeeves(fun, step=0.8, alpha=0.18, epsilon=eps, long_criterion=False, max_iterations=max_iter, print_stats=True) 
    graph.graph_search(points, fun)

    # Example of convergent search with same parameters, but also beta 
    points, iter, _ = hooke_jeeves_modified(fun, step=0.4, alpha=0.18, beta = 1.2, epsilon=eps, long_criterion=False, max_iterations=max_iter, print_stats=True) 
    graph.graph_search(points, fun)

    # 2.3.1
    # Обмеження
    constraints = [
        {'type': 'ineq', 'fun': lambda x, y: (x**2 + y**2 - 2.25)},  # x^2 + y^2 <= 2.25
        {'type': 'ineq', 'fun': lambda x, y: -(x**2 + y**2 - 0.25)}  # x^2 + y^2 >= 0.25
    ]

    # Початкові значення множників Лагранжа
    lambda_vals = [max(0, constraint['fun'](*fun.starting_point)) for constraint in constraints]

    # Параметри методу
    step = 0.4
    alpha = 0.18
    beta = 1.2
    epsilon = 1e-5
    max_iterations = 1000

    # Виклик модифікованого методу Хука-Дживса
    points, iterations, exceeded_max_iterations = hooke_jeeves_modified_with_constraints(
        fun, step, alpha, beta, epsilon, max_iterations=max_iterations,
        print_stats=True, constraints=constraints, lambda_vals=lambda_vals)

    graph.graph_search(points, fun, constraints=constraints)



    


    


