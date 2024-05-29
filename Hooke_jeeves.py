import copy
import numpy as np
from Functions import Function


def hooke_jeeves(fun: Function, 
                 step : float, 
                 alpha: float = 0.5, 
                 epsilon : float = 0.001,
                 long_criterion: bool = False, 
                 max_iterations: int = 1000, 
                 print_stats: bool = False):
    
    xb = copy.deepcopy(fun.starting_point)
    xp = copy.deepcopy(fun.starting_point)
    points = [copy.deepcopy(fun.starting_point)]
    
    fun.reset_iterations()
    
    while True:
        if long_criterion:
            # Ensure there are at least two points to compare
            if len(points) > 10:
                diff_vector = [points[-1][i] - points[-2][i] for i in range(2)]
                rel_change = np.linalg.norm(diff_vector) / np.linalg.norm(points[-2])
                func_change = abs(fun.calc(*points[-1]) - fun.calc(*points[-2]))
                fun.iterations -= 2
                if rel_change <= epsilon and func_change <= epsilon:
                    break
            elif fun.iterations > max_iterations:
                break
        else:
            if abs(step) <= epsilon:
                break
            elif fun.iterations > max_iterations:
                break
            
        xn = search(xp, fun, step)
        f_xn = fun.calc(*xn)
        f_xb = fun.calc(*xb)
        if f_xn < f_xb:
            for i in range(0, len(xn)):
                xp[i] = 2*xn[i] - xb[i]
                xb[i] = xn[i]
            points.append(copy.deepcopy(xp))
        else:
            step *= alpha
            xp = copy.deepcopy(xb)
            points.append(copy.deepcopy(xp))
        
    if print_stats:
        print(f"Final point: {points[-1]}")
        print(f"Final function value: {fun.calc(*points[-1])}")
        print(f"# iterations: {fun.iterations - 1}")

    return points, fun.iterations, fun.iterations > max_iterations


def search(xp, fun: Function, step):
    
    x = copy.deepcopy(xp)
    for i in range(0, len(xp)):
        p = fun.calc(*x)
        x[i] += step
        n = fun.calc(*x)
        if n > p:
            x[i] -= 2 * step
            n = fun.calc(*x)
            if n > p:
                x[i] += step
    return copy.deepcopy(x)



def hooke_jeeves_modified(fun: Function, 
                          step : float, 
                          alpha: float,
                          beta: float,  
                          epsilon : float = 0.001,
                          long_criterion: bool = False, 
                          max_iterations: int = 1000, 
                          print_stats: bool = False):
    
    xb = copy.deepcopy(fun.starting_point)
    xp = copy.deepcopy(fun.starting_point)
    points = [copy.deepcopy(fun.starting_point)]
    
    fun.reset_iterations()
    
    while True:
        if long_criterion:
            if len(points) > 10:
                diff_vector = [points[-1][i] - points[-2][i] for i in range(len(points[-1]))]
                rel_change = np.linalg.norm(diff_vector) / np.linalg.norm(points[-2])
                func_change = abs(fun.calc(*points[-1]) - fun.calc(*points[-2]))
                fun.iterations -= 2
                if rel_change <= epsilon and func_change <= epsilon:
                    break
        else:
            if abs(step) <= epsilon:
                break
        
        if fun.iterations > max_iterations:
            break
            
        xn = _search(xp, fun, step, beta)
        f_xn = fun.calc(*xn)
        f_xb = fun.calc(*xb)

        if f_xn < f_xb:
            for i in range(len(xn)):
                xp[i] = 2 * xn[i] - xb[i]
                xb[i] = xn[i]
            points.append(copy.deepcopy(xp))
        else:
            step *= alpha  # зменшити крок, якщо рух в неправильному напрямку
            xp = copy.deepcopy(xb)
            points.append(copy.deepcopy(xp))
        
    if print_stats:
        print(f"Final point: {points[-1]}")
        print(f"Final function value: {fun.calc(*points[-1])}")
        print(f"# iterations: {fun.iterations}")

    return points, fun.iterations, fun.iterations > max_iterations


def _search(xp, fun: Function, step, beta):
    x = copy.deepcopy(xp)
    for i in range(0, len(xp)):
        p = fun.calc(*x)
        x[i] += step
        n = fun.calc(*x)
        if n > p:
            x[i] -= 2 * beta * step
            n = fun.calc(*x)
            if n > p:
                x[i] += beta * step
    return copy.deepcopy(x)


def constraint_violations(x, constraints):
    violations = []
    for constraint in constraints:
        if constraint['type'] == 'eq':
            violations.append(abs(constraint['fun'](*x)))
        elif constraint['type'] == 'ineq':
            violations.append(max(0, constraint['fun'](*x)))
    return violations


def search_with_constraints(xp, fun: Function, step, beta, constraints, lambda_vals):
    x = copy.deepcopy(xp)
    n_vars = len(xp)
    n_constraints = len(constraints)
    
    for i in range(n_vars):
        x[i] += step
        violations = constraint_violations(x, constraints)
        lagrangian = fun.calc(*x) + sum(lam * viol for lam, viol in zip(lambda_vals, violations))
        
        x[i] -= step
        p = fun.calc(*x) + sum(lam * viol for lam, viol in zip(lambda_vals, constraint_violations(x, constraints)))
        
        if lagrangian < p:
            x[i] += step
        else:
            x[i] -= 2 * beta * step
            violations = constraint_violations(x, constraints)
            lagrangian = fun.calc(*x) + sum(lam * viol for lam, viol in zip(lambda_vals, violations))
            
            if lagrangian > p:
                x[i] += beta * step
                
    return copy.deepcopy(x)


def hooke_jeeves_modified_with_constraints(fun: Function, step: float, alpha: float, beta: float, epsilon: float = 0.001, long_criterion: bool = False, max_iterations: int = 1000, print_stats: bool = False, constraints=None, lambda_vals=None):
    xb = copy.deepcopy(fun.starting_point)
    xp = copy.deepcopy(fun.starting_point)
    points = [copy.deepcopy(fun.starting_point)]
    fun.reset_iterations()

    if constraints is None:
        constraints = []
    if lambda_vals is None:
        lambda_vals = [0.0] * len(constraints)

    while True:
        if long_criterion:
            if len(points) > 10:
                diff_vector = [points[-1][i] - points[-2][i] for i in range(len(points[-1]))]
                rel_change = np.linalg.norm(diff_vector) / np.linalg.norm(points[-2])
                func_change = abs(fun.calc(*points[-1]) - fun.calc(*points[-2]))
                fun.iterations -= 2
                if rel_change <= epsilon and func_change <= epsilon:
                    break

        else:
            if abs(step) <= epsilon:
                break

        if fun.iterations > max_iterations:
            break

        xn = search_with_constraints(xp, fun, step, beta, constraints, lambda_vals)
        violations = constraint_violations(xn, constraints)
        lagrangian = fun.calc(*xn) + sum(lam * viol for lam, viol in zip(lambda_vals, violations))

        f_xb = fun.calc(*xb) + sum(lam * viol for lam, viol in zip(lambda_vals, constraint_violations(xb, constraints)))

        if lagrangian < f_xb:
            for i in range(len(xn)):
                xp[i] = 2 * xn[i] - xb[i]
                xb[i] = xn[i]
            points.append(copy.deepcopy(xp))

            # Оновлення множників Лагранжа
            for j in range(len(lambda_vals)):
                lambda_vals[j] += step * violations[j]

        else:
            step *= alpha  # зменшити крок, якщо рух в неправильному напрямку
            xp = copy.deepcopy(xb)
            points.append(copy.deepcopy(xp))

            # Оновлення множників Лагранжа
            for j in range(len(lambda_vals)):
                lambda_vals[j] -= step * constraint_violations(xb, constraints)[j]

    if print_stats:
        print(f"Final point: {points[-1]}")
        print(f"Final function value: {fun.calc(*points[-1])}")
        print(f"# iterations: {fun.iterations}")

    return points, fun.iterations, fun.iterations > max_iterations



