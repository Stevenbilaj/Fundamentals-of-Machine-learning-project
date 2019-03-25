import numpy as np
import random
from random import sample
from random import shuffle
from settings import e
w_1 = np.load('agent_code/my_agent/first_weights.npy')
w_2 = np.load('agent_code/my_agent/second_weights.npy')
b_1 = np.load('agent_code/my_agent/first_biases.npy')
b_2 = np.load('agent_code/my_agent/second_biases.npy')


def setup(self):
    self.current_out = 0
    self.prev_state = np.zeros(450)
    self.next_state = np.zeros(450)
    self.reward = 0
    self.set = [self.reward, self.current_out, self.prev_state, self.next_state]
    self.Q_Network = QNeuralNetwork(w_1, w_2, b_1, b_2)
    self.Target_Network = QNeuralNetwork(self.Q_Network.first_weights, self.Q_Network.second_weights,
                                         self.Q_Network.first_biases, self.Q_Network.second_biases)
    self.memory = []
    self.C = 0
    self.epsil = 0.5
    return 0


def act(self):
    game_state = self.game_state
    inp = game_state_inp_alt(game_state)
    self.prev_state = inp
    q = self.Q_Network.forward_propagation(inp)
    Q = np.sum(np.exp(q))
    ran = random.random()
    if ran < np.exp(q[0]) / Q:
        self.next_action = 'LEFT'
        a = 0
    elif (ran >= np.exp(q)[0] / Q) and (ran < np.sum(np.exp(q)[:2]) / Q):
        self.next_action = 'RIGHT'
        a = 1
    elif (ran >= np.sum(np.exp(q)[:2]) / Q) and (ran < np.sum(np.exp(q)[:3]) / Q):
        self.next_action = 'UP'
        a = 2
    elif (ran >= np.sum(np.exp(q)[:3]) / Q) and (ran < np.sum(np.exp(q)[:4]) / Q):
        self.next_action = 'DOWN'
        a = 3
    elif (ran >= np.sum(np.exp(q)[:4]) / Q) and (ran < np.sum(np.exp(q)[:5]) / Q):
        self.next_action = 'BOMB'
        a = 4
    elif (ran >= np.sum(np.exp(q)[:5]) / Q) and (ran < 1):
        self.next_action = 'WAIT'
        a = 5
    # if random.random() < 1:
    #     q = self.Q_Network.forward_propagation(inp)
    #     a = np.argmax(q)
    #     if a == 0:
    #         self.next_action = 'LEFT'
    #     elif a == 1:
    #         self.next_action = 'RIGHT'
    #     elif a == 2:
    #         self.next_action = 'UP'
    #     elif a == 3:
    #         self.next_action = 'DOWN'
    #     elif a == 4:
    #         self.next_action = 'BOMB'
    #     else:
    #         self.next_action = 'WAIT'
    # else:
    #     a = np.random.randint(6, size=1)[0]
    #     if a == 0:
    #         self.next_action = 'LEFT'
    #     elif a == 1:
    #         self.next_action = 'RIGHT'
    #     elif a == 2:
    #         self.next_action = 'UP'
    #     elif a == 3:
    #         self.next_action = 'DOWN'
    #     elif a == 4:
    #         self.next_action = 'BOMB'
    #     else:
    #         self.next_action = 'WAIT'
    self.current_out = a
    return self.next_action


def game_state_inp(game_state):
    A = game_state['arena']
    inp = np.ravel(A[1:-1, 1:-1])
    B = game_state['self']
    B_0 = np.zeros((len(A) - 2, len(A) - 2))
    B_0[B[0] - 1][B[1] - 1] = 1
    inp = np.append(inp, np.ravel(B_0))
    # C = game_state['others']
    # C_0 = np.zeros((len(A) - 2, len(A) - 2))
    # for i in range(0, len(C)):
    #     C_0[C[i][0] - 1][C[i][1] - 1] = -1 + 2 * C[i][3]
    # inp = np.append(inp, C_0)
    D = game_state['bombs']
    D_0 = np.zeros((len(A) - 2, len(A) - 2))
    for i in range(0, len(D)):
        D_0[D[i][0] - 1][D[i][1] - 1] = 1/(D[i][2] + 1)
    inp = np.append(inp, D_0)
    E = game_state['coins']
    E_0 = np.zeros((len(A) - 2, len(A) - 2))
    for i in range(0, len(E)):
        E_0[E[i][0] - 1][E[i][1] - 1] = 4
    inp = np.append(inp, E_0)
    F = game_state['explosions']
    inp = np.append(inp, np.ravel(F[:-1, :-1][1:, 1:]))
    return inp


def game_state_inp_alt(game_state):
    A = game_state['arena']
    A = A[1:-1, 1:-1]
    E = game_state['coins']
    for i in range(0, len(E)):
        A[E[i][0] - 1][E[i][1] - 1] = 2
    D = game_state['bombs']
    for i in range(0, len(D)):
        A[D[i][0] - 1][D[i][1] - 1] = -1 - 1 / (D[i][2] + 1)
    F = game_state['explosions']
    F = F[1:-1, 1:-1]
    for i in range(0, len(F)):
        for j in range(0, len(F[0])):
            if F[i][j] != 0:
                A[i][j] = -2
    B = game_state['self']
    B_0 = np.zeros((len(A), len(A)))
    B_0[B[0] - 1][B[1] - 1] = 1 + B[3]
    A = np.ravel(A)
    B_0 = np.ravel(B_0)
    A = np.append(A, B_0)
    return A / 4


'''
Training Algorithm takes place here
'''


def reward_update(self):
    self.next_state = game_state_inp_alt(self.game_state)
    event = self.events
    for i in event:
        if i == 0:
            self.reward += -1
        elif i == 1:
            self.reward += -1
        elif i == 2:
            self.reward += -1
        elif i == 3:
            self.reward += -1
        elif i == 4:
            self.reward += -1
        elif i == 5:
            self.reward += -20
        elif i == 6:
            self.reward += -10
        elif i == 7:
            self.reward += -2
        elif i == 8:
            self.reward += 0
        elif i == 9:
            self.reward += 2
        elif i == 10:
            self.reward += 3
        elif i == 11:
            self.reward += 5
        elif i == 12:
            self.reward += 20
        elif i == 13:
            self.reward += -50
        elif i == 14:
            self.reward += -50
        elif i == 15:
            self.reward += 50
        elif i == 16:
            self.reward += 5
    if len(self.memory) == 320:
        self.memory = self.memory[1:]
    if self.game_state['step'] > 1:
        self.memory.append([self.reward, self.current_out, self.prev_state, self.next_state])
    self.reward = 0
    return 0


def end_of_episode(self):
    self.next_state = [0]
    event = self.events
    if self.epsil < 0.75:
        self.epsil += 0.005
    for i in event:
        if i == 0:
            self.reward += -1
        elif i == 1:
            self.reward += -1
        elif i == 2:
            self.reward += -1
        elif i == 3:
            self.reward += -1
        elif i == 4:
            self.reward += -2
        elif i == 5:
            self.reward += -20
        elif i == 6:
            self.reward += -10
        elif i == 7:
            self.reward += -1
        elif i == 8:
            self.reward += 0
        elif i == 9:
            self.reward += 2
        elif i == 10:
            self.reward += 3
        elif i == 11:
            self.reward += 5
        elif i == 12:
            self.reward += 20
        elif i == 13:
            self.reward += -50
        elif i == 14:
            self.reward += -50
        elif i == 15:
            self.reward += 10
        elif i == 16:
            self.reward += 5
    self.memory.append([self.reward, self.current_out, self.prev_state, self.next_state])
    self.reward = 0
    b_size = 32
    if len(self.memory) > b_size * 2:
        a = len(self.memory)
        N = np.trunc(a/b_size)
        N = N.astype(np.int64)
        for i in range(0, 2 * N):
            mini_batch = sample(self.memory, b_size)
            self.Q_Network.back_prop(mini_batch, self.Target_Network)
            self.C += 1
            if self.C == 1000:
                self.Target_Network = QNeuralNetwork(self.Q_Network.first_weights, self.Q_Network.second_weights,
                                                     self.Q_Network.first_biases, self.Q_Network.second_biases)
                self.C = 0
    np.save('agent_code/my_agent/first_weights', self.Q_Network.first_weights)
    np.save('agent_code/my_agent/second_weights', self.Q_Network.second_weights)
    np.save('agent_code/my_agent/first_biases', self.Q_Network.first_biases)
    np.save('agent_code/my_agent/second_biases', self.Q_Network.second_biases)
    return 0


class QNeuralNetwork:

    input_len = 450
    output_len = 6
    hidden_layer_count = 228
    gamma = 0.1
    learning_rate = 0.00001

    def __init__(self, weights_1=np.random.rand(hidden_layer_count, input_len) - 0.5,
                 weights_2=np.random.rand(6, hidden_layer_count) - 0.5, biases_1=np.zeros(hidden_layer_count),
                 biases_2=np.zeros(6)):
        self.first_weights = weights_1
        self.second_weights = weights_2
        self.first_biases = biases_1
        self.second_biases = biases_2
        self.In = np.zeros(self.input_len)
        self.H = np.zeros(self.hidden_layer_count)
        self.A_H = np.zeros(self.hidden_layer_count)
        self.Ou = np.zeros(self.output_len)
        self.A_O = np.zeros(self.output_len)
        self.prev_input = np.zeros(self.input_len)

    def forward_propagation(self, inp):
        self.prev_input = inp
        self.In = np.tanh(inp)
        self.H = np.dot(self.first_weights, self.In) + self.first_biases
        self.A_H = tanh_act(self.H)
        self.Ou = np.dot(self.second_weights, self.A_H) + self.second_biases
        self.A_O = tanh_act(self.Ou)
        return self.A_O

    def back_prop(self, mini_batch, target):
        corr = self.q_loss(mini_batch, target)
        self.first_weights = self.first_weights - self.learning_rate * np.reshape(corr[0], (self.hidden_layer_count, self.input_len))
        self.second_weights = self.second_weights - self.learning_rate * np.reshape(corr[1], (self.output_len, self.hidden_layer_count))
        self.first_biases = self.first_biases - self.learning_rate * corr[2]
        self.second_biases = self.second_biases - self.learning_rate * corr[3]
        return 0

    def q_loss(self, mini_batch, target):
        batch_size = len(mini_batch)
        correction_1 = np.zeros(self.input_len * self.hidden_layer_count)
        correction_2 = np.zeros(self.hidden_layer_count * self.output_len)
        correction_3 = np.zeros(self.hidden_layer_count)
        correction_4 = np.zeros(self.output_len)
        for i in range(0, batch_size):
            if np.sum(mini_batch[i][3]) == 0:
                prev_q = self.forward_propagation(mini_batch[i][2])
                q_gradi = self.q_gradient(mini_batch[i][2], mini_batch[i][1])
                correction_1 += ((mini_batch[i][0] - prev_q[mini_batch[i][1]])
                                 * q_gradi[0]) / batch_size
                correction_2 += ((mini_batch[i][0] - prev_q[mini_batch[i][1]])
                                 * q_gradi[1]) / batch_size
                correction_3 += ((mini_batch[i][0] - prev_q[mini_batch[i][1]])
                                 * q_gradi[2]) / batch_size
                correction_4 += ((mini_batch[i][0] - prev_q[mini_batch[i][1]])
                                 * q_gradi[3]) / batch_size
            else:
                next_q = target.forward_propagation(mini_batch[i][3])
                prev_q = self.forward_propagation(mini_batch[i][2])
                q_gradi = self.q_gradient(mini_batch[i][2], mini_batch[i][1])
                correction_1 += ((mini_batch[i][0] + self.gamma * np.max(next_q) - prev_q[mini_batch[i][1]])
                                 * q_gradi[0]) / batch_size
                correction_2 += ((mini_batch[i][0] + self.gamma * np.max(next_q) - prev_q[mini_batch[i][1]])
                                 * q_gradi[1]) / batch_size
                correction_3 += ((mini_batch[i][0] + self.gamma * np.max(next_q) - prev_q[mini_batch[i][1]])
                                 * q_gradi[2]) / batch_size
                correction_4 += ((mini_batch[i][0] + self.gamma * np.max(next_q) - prev_q[mini_batch[i][1]])
                                 * q_gradi[3]) / batch_size
        return correction_1, correction_2, correction_3, correction_4

    def q_gradient(self, inp, a):
        q_grad_2 = np.zeros((self.output_len, self.hidden_layer_count))
        q_grad_4 = np.zeros(self.output_len)
        self.forward_propagation(inp)
        q_grad_1 = np.ravel(der_tanh_act(self.Ou[a]) * np.outer(self.second_weights[a] * der_tanh_act(self.H), self.In))
        q_grad_2[a] = der_tanh_act(self.Ou[a]) * self.A_H
        q_grad_2 = np.ravel(q_grad_2)
        q_grad_3 = der_tanh_act(self.Ou[a]) * self.second_weights[a] * der_tanh_act(self.H)
        q_grad_4[a] = der_tanh_act(self.Ou[a])
        return q_grad_1, q_grad_2, q_grad_3, q_grad_4


def activation_function_(x):
    act = 1/(1 + np.exp(-x))
    return act


def activation_function(x):
    x = np.maximum(x, 0)
    return x


def der_activation_func_(x):
    act = activation_function_(x) * (1 - activation_function_(x))
    return act


def der_activation_func(x):
    r = np.heaviside(x, 0)
    return r


def tanh_act(x):
    return np.tanh(x)


def der_tanh_act(x):
    return 1/(np.cosh(x)**2)
