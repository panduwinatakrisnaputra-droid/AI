# Modified Q_test.py for an 8-State Routing Problem
import numpy as np
import matplotlib.pyplot as plt
from Q_Utils import *

# 1. Routing edges for states 0-7 (Goal = 7)
points_list = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 5), (5, 6), (6, 7)]
goal = 7

# Display routing graph
showgraph(points_list)

# 2. Number of states
MATRIX_SIZE = 8

# Create R and Q matrices
R = createRmat(MATRIX_SIZE, points_list, goal)
Q = np.matrix(np.zeros([MATRIX_SIZE, MATRIX_SIZE]))
gamma = 0.8

# 3. Training Loop
scores = []
for i in range(700):
    current_state = np.random.randint(0, int(Q.shape[0]))
    available_act = available_actions(R, current_state)
    action = sample_next_action(available_act)
    score = update(R, Q, current_state, action, gamma)
    scores.append(score)

print("Trained Q matrix:")
print(Q / np.max(Q) * 100)

# 4. Testing Phase
current_state = 0
steps = [current_state]
while current_state != goal:
    next_step_index = np.where(Q[current_state,] == np.max(Q[current_state,]))[1]
    if next_step_index.shape[0] > 1:
        next_step_index = int(np.random.choice(next_step_index, size=1))
    else:
        next_step_index = int(next_step_index)
    steps.append(next_step_index)
    current_state = next_step_index

print("Most efficient path:", steps)
plt.plot(scores)
plt.show()