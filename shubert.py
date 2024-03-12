import matplotlib.pyplot as plt
import numpy as np

# AT LEAST 2 ITERATIONS
MAX_ITER = 100

'''Utilities:'''


def insert_in_order(lista: list, x_value, z_value) -> None:
    if lista[0][0] > x_value:
        lista.insert(0, (x_value, z_value))
        return
    for i in range(len(lista) - 1):
        if lista[i][0] < x_value < lista[i + 1][0]:
            lista.insert(i + 1, (x_value, z_value))
            return
    lista.append((x_value, z_value))
    return


def calculate_r(xz: list, i: int, lip) -> float:
    # i is the index of the selected range.
    # for example: the range [x0, x1] will have index 0.
    # generally: the range [xn, xn+1] will have index n.
    return (xz[i][1] + xz[i + 1][1]) / 2 - lip * (xz[i + 1][0] - xz[i][0]) / 2


def index_min(l: list) -> int:
    min_current = l[0]
    index_min_current = 0
    for i in range(len(l)):
        if l[i] < min_current:
            min_current = l[i]
            index_min_current = i
    return index_min_current


'''End of utilities'''


def evaluate_min(f, a: float, b: float, lip: float, e: float, gui=True, plot_accuracy=0.05) -> float:
    # estimate the global minimum value of the f function, having the actual Lipschitz constant < lip,
    # using the Piyavskii–Shubert Algorithm
    if gui:
        X = np.arange(a, b, plot_accuracy)
        plt.plot(X, f(X), color='b', label='f')
    k = 1
    # first iteration is done out of the loop.
    #   history = [a, b]
    xz = [(a, f(a)), (b, f(b))]
    # list of Tuples. Contains the coords of each evaluated point, sorted by x
    # generic access: xz[i][0] access to the value of xi
    # xz[i][1] access to the value of f(xi) (also called zi)
    r = [((xz[0][1] + xz[1][1]) / 2 - lip * (xz[1][0] - xz[0][0]) / 2)]
    # r = [r1]
    if xz[1][0] - xz[0][0] < e:
        return min(xz[0][1], xz[1][1])
    xt = (
                (xz[0][0] + xz[1][0]) / 2 - (xz[1][1] - xz[0][1]) / (2 * lip))
    #   history.append(x_current_min)
    zt = f(xt)
    insert_in_order(xz, xt, zt)
    k += 1
    selected_range = 0
    x_current_min = xt
    current_min = zt
    for i in range(len(xz)):
        if xz[i][1] < current_min:
            current_min = xz[i][1]
            x_current_min = xz[i][0]

    while k < MAX_ITER:
        # step 1: should re-order the xz list, but it's already sorted.
        # step 2: evaluating the Ri values
        # N.B: evaluating only the values of the new 2 obtained ranges is fine, other values will not change.
        r[selected_range] = calculate_r(xz, selected_range, lip)
        r.insert(selected_range + 1, calculate_r(xz, selected_range + 1, lip))
        # step 3: looking for the range with the lowest value of Ri.
        selected_range = index_min(r)
        # step 4: exit criterion
        if xz[selected_range+1][0] - xz[selected_range][0] < e:
            # e (epsilon) is the 'accuracy' desired.
            break
        # step 5: evaluating f(x) on the xt point corresponding to the Ri value
        xt = (xz[selected_range][0]+xz[selected_range+1][0])/2 - (xz[selected_range+1][1]-xz[selected_range][1])/(2*lip)
        #   history.append(xt)
        zt = f(xt)
        if zt < current_min:
            current_min = zt
            x_current_min = xt
        insert_in_order(xz, xt, zt)
        k += 1
    if k >= MAX_ITER:
        print("MAX ITER limit exceeded. Quitting.")
        r[selected_range] = calculate_r(xz, selected_range, lip)
        r.insert(selected_range + 1, calculate_r(xz, selected_range + 1, lip))
    if gui:
        x = []
        y = []
        for i in range(len(xz)-1):
            xt = (xz[i][0] + xz[i + 1][0]) / 2 - (
                        xz[i + 1][1] - xz[i][1]) / (2 * lip)
            x.append(xz[i][0])
            x.append(xt)
            y.append(xz[i][1])
            y.append(r[i])
            #   plt.axvline(x=xz[i+1][0],ymin=0.5, ymax=0.7)
        x.append(xz[len(xz)-1][0])
        y.append(xz[len(xz)-1][1])
        plt.plot(x, y, 'ro--', label='lower bound')
        plt.plot(x_current_min, current_min, color='violet', marker='D', linestyle='-', label='global minimum')
        plt.legend()
        plt.show()
    return current_min
