import matplotlib.pyplot as plt
import numpy as np
from Hooke_jeeves import hooke_jeeves, hooke_jeeves_modified
from Functions import Function


def graph_search(_points_, _fun_: Function, constraints=None):
    x, y = zip(*_points_)
    x = list(x)
    y = list(y)

    plt.figure(figsize=(12, 8))

    x_grid = np.linspace(-1.8, 2, 500)
    y_grid = np.linspace(-1.8, 2, 500)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = _fun_.calc(*np.meshgrid(x_grid, y_grid))
    Z = np.log(Z + 0.0001)

    contour = plt.contour(X, Y, Z, levels=20, cmap='viridis', linewidths=1)
    plt.clabel(contour, inline=1, fontsize=8)

    # Візуалізація обмежень рівності
    if constraints is not None:
        for constraint in constraints:
            if constraint['type'] == 'eq':
                eq_vals = constraint['fun'](X, Y)
                plt.contour(X, Y, eq_vals, levels=[0], colors='black', linewidths=1.5, linestyles='dashed')

        # Візуалізація обмежень нерівності
        for constraint in constraints:
            if constraint['type'] == 'ineq':
                ineq_vals = constraint['fun'](X, Y)
                plt.contour(X, Y, ineq_vals, levels=[0], colors='green', linewidths=1.5, linestyles='dashed')
                plt.contourf(X, Y, ineq_vals, levels=[-10, 0], colors='green', alpha=0.2)

    for i in range(len(_points_) - 1):
        plt.arrow(_points_[i][0], _points_[i][1],
                  _points_[i + 1][0] - _points_[i][0], _points_[i + 1][1] - _points_[i][1],
                  head_width=0.0001, head_length=0.0001,
                  fc='red', ec='red')

    sizes = np.linspace(2000, 100, len(_points_))
    for i in range(len(_points_) - 1):
        plt.scatter(x[i], y[i], color='blue', s=sizes[i] ** 0.4)

    plt.scatter(x[-1], y[-1], color='magenta', s=20, label='Знайдена точка мінімуму')
    plt.scatter(_fun_.starting_point[0], _fun_.starting_point[1], color='red', s=20, label='Початкова точка')
    plt.scatter(_fun_.point_min[0], _fun_.point_min[1], color='orange', s=10, label='Справжній мінімум')

    plt.axhline(0, color='black', linestyle='--', lw=0.75)
    plt.axvline(0, color='black', linestyle='--', lw=0.75)

    plt.legend()
    plt.xlabel("X1")
    plt.ylabel("X2", rotation=0)
    plt.show()


def graph_iterations_num_of_dx(fun: Function, 
                               min_step: float, 
                               max_step: float, 
                               delta: float,
                               alpha: float,
                               epsilon: float, 
                               max_iterations: int):
    # Define range for dx
    step_values = list(np.arange(min_step, max_step, delta))
    num_calculations = []
    filtered_step_values = []

    for step in step_values:
        _, iter, _ = hooke_jeeves(fun, step, alpha, epsilon=epsilon, long_criterion=False, max_iterations=max_iterations)
        if iter <= max_iterations:
            filtered_step_values.append(step)
            num_calculations.append(iter)
    
    # Find the optimal step
    min_iterations = min(num_calculations)
    optimal_step_index = num_calculations.index(min_iterations)
    optimal_step = filtered_step_values[optimal_step_index]

    plt.figure(figsize=(12, 5))
    plt.plot(filtered_step_values, num_calculations, label=f"alpha = {alpha}")

    for step in filtered_step_values[::2]:
        plt.axvline(x=step, color='gray', linestyle='--', linewidth=0.5)
    
    # Plot the optimal step
    plt.scatter([optimal_step], [min_iterations], color='orange', zorder=5)
    plt.text(optimal_step+0.04, min_iterations-5, f'Оптимальний крок Δx = {optimal_step:.2f}', fontsize=10, verticalalignment='bottom')

    # Set X-axis ticks every fifth value
    plt.xticks(filtered_step_values[::5])

    # Plotting the results
    plt.xlabel('Початковий крок Δx')
    plt.ylabel('Кількість обчислень функції')
    plt.title('Залежність швидкості збіжності від значення початкового кроку')
    plt.grid(True)
    plt.legend()
    plt.show()


def graph_iterations_num_of_alpha(fun: Function, 
                                  step: float,
                                  min_alpha: float, 
                                  max_alpha: float, 
                                  delta: float,
                                  epsilon: float, 
                                  max_iterations: int):
    # Define range for alpha
    alpha_values = list(np.arange(min_alpha, max_alpha, delta))
    filtered_alpha_values = []
    num_calculations = []

    # Run Hooke-Jeeves for each alpha and record number of function evaluations
    for alpha in alpha_values:
        _, iter, reached = hooke_jeeves(fun, step, alpha, epsilon=epsilon, long_criterion=False, max_iterations=max_iterations)
        if reached:
            filtered_alpha_values.append(alpha)
            num_calculations.append(iter)

    
    plt.figure(figsize=(12, 6))
    plt.plot(filtered_alpha_values, num_calculations, label=f"Початковий крок = {step}")

    for alpha in filtered_alpha_values[:]:
        plt.axvline(x=alpha, color='gray', linestyle='--', linewidth=0.5)
    
    # Find the optimal alpha
    min_iterations = min(num_calculations)
    optimal_alpha_index = num_calculations.index(min_iterations)
    optimal_alpha = filtered_alpha_values[optimal_alpha_index]

    # Plot the optimal alpha
    plt.scatter([optimal_alpha], [min_iterations], color='orange', zorder=5)
    plt.text(optimal_alpha+0.004, min_iterations-5, f'Оптимальне α = {optimal_alpha:.2f}', fontsize=10, verticalalignment='bottom')

    # Plotting the results
    plt.xlabel('Значення альфа α')
    plt.ylabel('Кількість обчислень функції')
    plt.title('Залежність швидкості збіжності від значення α (коефіцієнта зменшення кроку)')
    plt.grid(True)
    plt.legend()
    plt.show()



def graph_long_short_criterion_fixed_alpha(fun: Function, 
                                           steps: list,
                                           alpha: float,
                                           epsilon: float, 
                                           max_iterations: int):
    long_iterations = []
    short_iterations = []
    steps_list_short = steps.copy()  # Steps list for the short criterion
    
    for step in steps:
        _, iter, reached = hooke_jeeves(fun, step, alpha, epsilon, long_criterion=False, max_iterations=max_iterations)
        if reached:
            steps_list_short.remove(step)
        else:
            short_iterations.append(iter)
    
    for step in steps:
        if step in steps_list_short:
            _, iter, _ = hooke_jeeves(fun, step, alpha, epsilon, long_criterion=True, max_iterations=max_iterations)
            long_iterations.append(iter)

    # Adjust step values for the short criterion to match the length of iterations lists
    steps_list_short = [step for step in steps if step in steps_list_short]
    
    plt.figure(figsize=(12, 6))
    plt.plot(steps_list_short, short_iterations, label='Критерій |Δx| < ε', color='orange')
    plt.plot(steps_list_short, long_iterations, label='Довгий критерій', color='blue')
    plt.xlabel('Розмір початкового кроку')
    plt.ylabel('Кількість обчислень цільової функції')
    plt.title(f'Залежність швидкості збіжності від критерію зупинки алгоритму (alpha = {alpha})')
    plt.legend()
    plt.grid(True)

    # Adding vertical lines for each step in steps_list_short
    for step in steps_list_short:
        plt.axvline(x=step, color='grey', linestyle='--', linewidth=0.25)

    plt.show()


def graph_long_short_criterion_fixed_step(fun: Function, 
                                          step: float,
                                          alphas: list,
                                          epsilon: float, 
                                          max_iterations: int):
    long_iterations = []
    short_iterations = []
    alpha_list_short = alphas.copy()  # alpha list for the short criterion
    
    for alpha in alphas:
        _, iter, reached = hooke_jeeves(fun, step, alpha, epsilon, long_criterion=False, max_iterations=max_iterations)
        if reached:
            alpha_list_short.remove(alpha)
        else:
            short_iterations.append(iter)
    
    for alpha in alphas:
        if alpha in alpha_list_short:
            _, iter, _ = hooke_jeeves(fun, step, alpha, epsilon, long_criterion=True, max_iterations=max_iterations)
            long_iterations.append(iter)

    # Adjust alpha values for the short criterion to match the length of iterations lists
    alpha_list_short = [alpha for alpha in alphas if alpha in alpha_list_short]
    
    plt.figure(figsize=(12, 6))
    plt.plot(alpha_list_short, short_iterations, label='Критерій |Δx| < ε', color='orange')
    plt.plot(alpha_list_short, long_iterations, label='Довгий критерій', color='blue')
    plt.xlabel('Значення альфа α')
    plt.ylabel('Кількість обчислень цільової функції')
    plt.title(f'Залежність швидкості збіжності від критерію зупинки алгоритму (початковий крок = {step})')
    plt.legend()
    plt.grid(True)

    # Adding vertical lines for each step in alpha_list_short
    for alpha in alpha_list_short:
        plt.axvline(x=alpha, color='grey', linestyle='--', linewidth=0.25)

    plt.show()


def graph_beta_difference_fixed_alpha(fun: Function, 
                                      steps: list,
                                      alpha: float,
                                      beta: float,
                                      epsilon: float, 
                                      max_iterations: int):
    alpha_iterations = []
    beta_iterations = []
    filtered_step_values = steps.copy()
    
    for step in steps:
        _, iter, reached = hooke_jeeves(fun, step, alpha, epsilon, max_iterations=max_iterations)
        if reached:
            filtered_step_values.remove(step)
        else:
            alpha_iterations.append(iter)
    
    for step in steps:
        if step in filtered_step_values:
            _, iter, _ = hooke_jeeves_modified(fun, step, alpha, beta=beta, epsilon=epsilon, max_iterations=max_iterations)
            beta_iterations.append(iter)


    # Adjust alpha values for the short criterion to match the length of iterations lists
    steps_list_short = [step for step in steps if step in filtered_step_values]

    plt.figure(figsize=(13, 5))
    plt.plot(steps_list_short, alpha_iterations, label='Оригінальний алгоритм', color='orange')
    plt.plot(steps_list_short, beta_iterations, label=f'beta = {beta}', color='blue')
    plt.xlabel('Значення початкового кроку')
    plt.ylabel('Кількість обчислень цільової функції')
    plt.title(f'Залежність швидкості збіжності від модифікації алгоритму (alpha = {alpha})')
    plt.legend()
    plt.grid(True)

    # Adding vertical lines for each step in steps_list_short
    for s in steps_list_short:
        plt.axvline(x=s, color='grey', linestyle='--', linewidth=0.25)

    plt.show()


def graph_beta_difference_fixed_alpha_all(fun: Function, 
                                      steps: list,
                                      alpha: float,
                                      beta: float,
                                      epsilon: float, 
                                      max_iterations: int):
    alpha_iterations = []
    beta_iterations = []
    
    for step in steps:
        _, iter, _ = hooke_jeeves(fun, step, alpha, epsilon, max_iterations=max_iterations)
        alpha_iterations.append(iter)
    
    for step in steps:
        _, iter, _ = hooke_jeeves_modified(fun, step, alpha, beta=beta, epsilon=epsilon, max_iterations=max_iterations)
        beta_iterations.append(iter)

    plt.figure(figsize=(12, 6))
    plt.plot(steps, alpha_iterations, label='Оригінальний алгоритм', color='orange')
    plt.plot(steps, beta_iterations, label=f'beta = {beta}', color='blue')
    plt.xlabel('Значення початкового кроку')
    plt.ylabel('Кількість обчислень цільової функції')
    plt.title(f'Залежність швидкості збіжності від модифікації алгоритму (alpha = {alpha})')
    plt.legend()
    plt.grid(True)

    # Adding vertical lines for each step in steps_list_short
    for s in steps:
        plt.axvline(x=s, color='grey', linestyle='--', linewidth=0.2)

    plt.show()

        
