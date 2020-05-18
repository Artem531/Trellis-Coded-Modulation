import collections
from queue import Queue
import random
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np


# present/input zero/input one
Next_state_table = [[0, 1, 0, 1],
                    [2, 3, 2, 3],
                    [1, 0, 1, 0],
                    [3, 2, 3, 2]]



# present/input zero/input one
Output_table = [ [[0,0,0],[0,1,0], [1,0,0],[1,1,0]],
                 [[0,0,1],[0,1,1], [1,0,1],[1,1,1]],
                 [[0,0,0],[0,1,0], [1,0,0],[1,1,0]],
                 [[0,0,1],[0,1,1], [1,0,1],[1,1,1]] ]

sq2div2 = np.sqrt(2) / 2

# present/input zero/input one
Output_table_coords = [[sq2div2 - 1j * sq2div2, sq2div2 + 1j * sq2div2, -sq2div2 + 1j * sq2div2, -sq2div2 - 1j * sq2div2],
                       [1, 1j, -1, -1j],
                       [sq2div2 - 1j * sq2div2, sq2div2 + 1j * sq2div2, -sq2div2 + 1j * sq2div2, -sq2div2 - 1j * sq2div2],
                       [1, 1j, -1, -1j]
                      ]


def get_distance(a, b):
    return (a.real - b.real)**2 + (a.imag - b.imag)**2

def part_time_I_Viterbi(cur_dist, state, next_dist, val):
    """
    Viterbi for state in t=i
    :param cur_dist: current distances
    :param state: idx of current state
    :param next_dist: next state distances
    :param val: received signal
    :return:
    """
    state_idx = state
    for i, next_state_value in enumerate(Output_table_coords[state_idx]):
        dist = get_distance(next_state_value, val)
        state_next_state_dist = cur_dist[state_idx] + dist
        next_state_idx = Next_state_table[state_idx][i]

        if next_dist[next_state_idx][0] > state_next_state_dist:
            next_dist[next_state_idx][0] = state_next_state_dist
            next_dist[next_state_idx][1] = state_idx # prev state in history  #Where_i_was_table[next_state_idx][np.argmin(np.array(cur_dist)[Where_i_was_table[next_state_idx]])]
            next_dist[next_state_idx][3] = next_state_idx # cur state in history
            next_dist[next_state_idx][2] = i

    return next_dist

def full_time_I_Viterbi(cur_dist, val):
    """
    Viterbi for t=i
    :param cur_dist: current distances
    :param val: received signal
    :return:
    """
    next_dist = [[float('Inf'), -1, -1, -1],[float('Inf'), -1, -1, -1],[float('Inf'), -1, -1, -1],[float('Inf'), -1, -1,-1]]

    cur_states = np.array(range(4))[np.array(np.array(cur_dist) != float('Inf'))]

    for state_idx in cur_states:
        next_dist = part_time_I_Viterbi(cur_dist, state_idx, next_dist, val)
    return next_dist

def full_Viterbi(cur_dist, val_arr):
    """
    Main Viterbi function
    :param cur_dist: current distances
    :param val_arr: array with received signals
    :return:
    """
    next_history = []

    for val in val_arr: # get val coord
        next_dist = full_time_I_Viterbi(cur_dist, val)
        next_history.append(next_dist)
        next_dist = np.array(next_dist)
        cur_dist = list(next_dist[:, 0])
    return next_history, cur_dist

def get_vals(val_arr, start):
    res_val_arr = []
    for i in val_arr: # get val idx
        # get coords
        res_val_arr.append(Output_table_coords[start][i])
        # go to next state
        start = Next_state_table[start][i]
    return res_val_arr, start

def find(state, next_state):
    for i, tmp_next_state in enumerate(Next_state_table[state]):
        if tmp_next_state == next_state:
            return i
    return -1

def get_ans(history):
    """
    Get answer from trellis history
    :param history: trellis history
    :return: min path
    """
    res = []
    for i in reversed(history):
        i = np.array(i)
        idx = np.argmin(i[:, 0])
        where = idx
        break

    for i in reversed(range(1, len(history))):
        cur = np.array(history[i])
        prev = np.array(history[i-1])
        res.append(int(cur[where, 2]))
        where = np.argmax(cur[where, 1] == prev[:, 3])

    cur = np.array(history[0])
    res.append(int(cur[where, 2]))

    return list(reversed(res))

def stream():
    """
    modulation function
    :return: None
    """
    cur_dist = [0, float('Inf'), float('Inf'), float('Inf')]
    Maxdb = 11
    SNRdb = list(np.linspace(0, Maxdb, 5))

    Pe = []
    E = 1
    for db in SNRdb:
        SNR = 10 ** (db / 10)
        Nerr = 0
        Ntest = 0
        sigma = np.sqrt(E / (2 * SNR))

        num = 60
        Nerr_max = 50
        val_arr = random.choices(population=[0, 1, 2, 3], k=num)
        start = 0
        init = True
        while (Nerr < Nerr_max):

            print("send", val_arr)
            if init:
                vals, start = get_vals(val_arr, start)
            else:
                vals, start = get_vals([val_arr[-1]], start)
                vals[0] += 0j

            val_arr1 = np.array(vals)

            val_arr1 += 1j * np.random.normal(0, sigma, 1) + 1 * np.random.normal(0, sigma, 1)

            if init:
                history, cur_dist = full_Viterbi(cur_dist, val_arr1)
                init = False
            else:
                history1, cur_dist = full_Viterbi(cur_dist, val_arr1)
                history = history + history1
                history = history[1:]

            pred = get_ans(history)[0]

            print("reci", get_ans(history))
            print(Nerr, db, Ntest)
            if pred != val_arr[0]:
                print("Error!")
                Nerr += 1
            val_arr = val_arr + random.choices(population=[0, 1, 2, 3], k=1)
            val_arr = val_arr[1:]
            Ntest += 1
        Pe.append(Nerr/Ntest)


    SNRdbtheory = np.linspace(0, Maxdb + 5, 100)
    SNR = 10 ** (SNRdbtheory / 10)

    q = 4
    Q4 = 4*(np.sqrt(q)-1)/q * norm.sf(np.sqrt(3*SNR*1/(q-1)))*(np.sqrt(q) - ((np.sqrt(q)) - 1) * norm.sf(np.sqrt(3*SNR*1/(q-1))))
    q = 2
    Q2 = 2 * norm.sf(np.sqrt(2 * SNR) * np.sin(np.pi / q))


    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_yscale("log")
    ax1.set_ylabel('Pe')
    ax1.set_xlabel('SNRdb')

    plt.scatter(SNRdb, Pe, s=2, label="test")
    plt.plot(SNRdbtheory, Q4, label='acc 4')
    plt.plot(SNRdbtheory, Q2, label='acc 2')
    plt.legend()
    plt.show()

