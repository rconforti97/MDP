# Rachel Conforti 
# All you have to do is run the python file. 
# Please have numpy package installed 

import numpy as np

# Create a class for a State that has members ‘x’ and ‘y’.
class State:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# Create a 1D list of State objects for the environment. Exclude the state for position (2,2) b/c obstacle.
objects = [State(1, 1), State(2,1), State(3,1), State(4,1), State(1,2), State(3,2), State(4,2), State(1,3), State(2,3), State(3,3), State(4,3)]

# Create a list of actions
actions = ['U', 'R', 'D', 'L']

# Given transition model by professor
P = [[[0.1, 0.1, 0.,  0.,  0.8, 0.,  0.,  0.,  0.,  0.,  0.],
  [0.1, 0.8, 0.1, 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0. ],
  [0.,  0.1, 0.,  0.1, 0.,  0.8, 0.,  0.,  0.,  0.,  0. ],
  [0.,  0.,  0.1, 0.1, 0.,  0.,  0.8, 0.,  0.,  0.,  0. ],
  [0.,  0.,  0.,  0.,  0.2, 0.,  0.,  0.8, 0.,  0.,  0. ],
  [0.,  0.,  0.,  0.,  0.,  0.1, 0.1, 0.,  0.,  0.8, 0. ],
  [0.,  0.,  0.,  0.,  0.,  0.1, 0.1, 0.,  0.,  0.,  0.8],
  [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.9, 0.1, 0.,  0. ],
  [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.1, 0.8, 0.1, 0. ],
  [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.1, 0.8, 0.1],
  [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.1, 0.9]],

 [[0.1, 0.8, 0.,  0.,  0.1, 0.,  0.,  0.,  0.,  0.,  0. ],
  [0.,  0.2, 0.8, 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0. ],
  [0.,  0.,  0.1, 0.8, 0.,  0.1, 0.,  0.,  0.,  0.,  0. ],
  [0.,  0.,  0.,  0.9, 0.,  0.,  0.1, 0.,  0.,  0.,  0. ],
  [0.1, 0.,  0.,  0.,  0.8, 0.,  0.,  0.1, 0.,  0.,  0. ],
  [0.,  0.,  0.1, 0.,  0.,  0.,  0.8, 0.,  0.,  0.1, 0. ],
  [0.,  0.,  0.,  0.1, 0.,  0.,  0.8, 0.,  0.,  0.,  0.1],
  [0.,  0.,  0.,  0.,  0.1, 0.,  0.,  0.1, 0.8, 0.,  0. ],
  [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.2, 0.8, 0. ],
  [0.,  0.,  0.,  0.,  0.,  0.1, 0.,  0.,  0.,  0.1, 0.8],
  [0.,  0.,  0.,  0.,  0.,  0.,  0.1, 0.,  0.,  0.,  0.9]],

 [[0.9, 0.1, 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0. ],
  [0.1, 0.8, 0.1, 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0. ],
  [0.,  0.1, 0.8, 0.1, 0.,  0.,  0.,  0.,  0.,  0.,  0. ],
  [0.,  0.,  0.1, 0.9, 0.,  0.,  0.,  0.,  0.,  0.,  0. ],
  [0.8, 0.,  0.,  0.,  0.2, 0.,  0.,  0.,  0.,  0.,  0. ],
  [0.,  0.,  0.8, 0.,  0.,  0.1, 0.1, 0.,  0.,  0.,  0. ],
  [0.,  0.,  0.,  0.8, 0.,  0.1, 0.1, 0.,  0.,  0.,  0. ],
  [0.,  0.,  0.,  0.,  0.8, 0.,  0.,  0.1, 0.1, 0.,  0. ],
  [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.1, 0.8, 0.1, 0. ],
  [0.,  0.,  0.,  0.,  0.,  0.8, 0.,  0.,  0.1, 0.,  0.1],
  [0.,  0.,  0.,  0.,  0.,  0.,  0.8, 0.,  0.,  0.1, 0.1]],

 [[0.9, 0.,  0.,  0.,  0.1, 0.,  0.,  0.,  0.,  0.,  0. ],
  [0.8, 0.2, 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0. ],
  [0.,  0.8, 0.1, 0.,  0.,  0.1, 0.,  0.,  0.,  0.,  0. ],
  [0.,  0.,  0.8, 0.1, 0.,  0.,  0.1, 0.,  0.,  0.,  0. ],
  [0.1, 0.,  0.,  0.,  0.8, 0.,  0.,  0.1, 0.,  0.,  0. ],
  [0.,  0.,  0.1, 0.,  0.,  0.8, 0.,  0.,  0.,  0.1, 0. ],
  [0.,  0.,  0.,  0.1, 0.,  0.8, 0.,  0.,  0.,  0.,  0.1],
  [0.,  0.,  0.,  0.,  0.1, 0.,  0.,  0.9, 0.,  0.,  0. ],
  [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.8, 0.2, 0.,  0. ],
  [0.,  0.,  0.,  0.,  0.,  0.1, 0.,  0.,  0.8, 0.1, 0. ],
  [0.,  0.,  0.,  0.,  0.,  0.,  0.1, 0.,  0.,  0.8, 0.1]]]
P = np.array(P)

# think this is where I am going wrong or policy print. 
def getExpectedUtility(utilities, P, states, passed_in_position):
    # initalizing 
    choosen_action = 0
    # initalizing our testing point
    max_expected_utilities = -1
    # for each action
    for index in range(4):
        expected_utilities = 0
        # for the potential next state of our objects 
        for next_state in states:
            # getting u prine by getting the utility at the location of the next state
            u_prime = utilities[next_state]
            # getting the probability from the transition model 
            prob_of_next_state = P[index, passed_in_position, next_state]
            # adding the expected utility by multiplying the probability and u prime
            expected_utilities += (prob_of_next_state * u_prime)

            # This is to see if we hit the max expected utility
            # to ensure we are chosing the highest utility
            # if its higher then we add the action to the list
            if expected_utilities > max_expected_utilities:
                max_expected_utilities = expected_utilities
                choosen_action = index
        # return the list of actions to add to the policy 
        list_of_actions = actions[choosen_action]
    
    return list_of_actions

# This is the value iteration method 
def valueIteration(P, passed_in_rewards, passed_in_discount):
    delta = 0
    # this is our breaking point of the code
    threshold = 0.0001 * (1 - passed_in_discount) / passed_in_discount
    # Creating the utility list and setting terminal states
    U = np.zeros(11)
    U[6] = -1
    U[10] = 1
    # Creating u prime
    U_prime = np.zeros(11)
    # creating list of passed in reward states and hard coding terminal states
    r = np.full(11, passed_in_rewards) 
    r[6] = -1
    r[10] = 1

    while True:
        for i in range(11):
            # check if terminal state - if so move on
            if i == 6 or i == 10:
                continue
            act = np.zeros(4)
            # for each of the 4 possible actions
            for j in range(4):
                # check it against each state
                for k in range(11):
                    # part of the bellman equation  
                    act[j] += (U[k] * P[j][i][k])

            # u prime equals reward for this state plus the discount 
            # plus the highest action from above. This is the bellman equation
            U_prime[i] = r[i] + passed_in_discount * max(act)
            # taking absolute value b/c without it can be below zero automacatically
            delta = abs(U_prime[i] - U[i])
            U[i] = U_prime[i]
        
        # checking for break condition
        if delta < threshold:
            break
    
    print("Utilities:", U)
    return U

# Purpose of this method is to let us know what states are 
# valid for us to go to. AKA checking if we hit a boundry
def getValidStates(objects, state):
    valid_states = []
     
    # Get the index of neighbor in each direction
    # and check if its a valid move. Following the 
    # up, right, down, left cycle. 
    up = isNextStateValid(objects, state.x, state.y+1)
    if up != -1:
        valid_states.append(up)

    right = isNextStateValid(objects, state.x+1, state.y)
    if right != -1:
        valid_states.append(right)

    down = isNextStateValid(objects, state.x, state.y-1)
    if down != -1:
        valid_states.append(down)

    left = isNextStateValid(objects, state.x-1, state.y)
    if left != -1:
        valid_states.append(left)

    return valid_states

# this is a method used in get valid states - checking the index
# if it equals a valid state it returns index if not -1 to indicate 
# that its invalid
def isNextStateValid(states, x_coord, y_coord):
    for index, state_object in enumerate(states):
        if state_object.x == x_coord and state_object.y == y_coord:
            return index
    return -1

def printPolicy(utilities, P):
    policy = []

    for index, states in enumerate(objects):
        # Checking if we are entering a terminal state
        if index == 6 or index == 10:
            action = 'T'
        # Get the valid states and expect utilities which will choose our actions
        else:
            valid_states = getValidStates(objects, states)
            action = getExpectedUtility(utilities, P, valid_states, index)
            
        policy.append(action)

    # Inserting obstacle and printing them out section by section
    policy.insert(5, '0')
    print(policy[8:12])
    print(policy[4:8])
    print(policy[:4])

# To clean up my main method a bit, this is printing out the reward
# and discount, calling my value iteration and print policy method
def runMDP(P, reward, discount):
    print("Discount =", discount, " Reward =", reward)
    value_iteration = valueIteration(P, reward, discount)
    printPolicy( value_iteration, P)
    print("\n###################################################################################\n")

# calling the 3 examples from the assignment 
def main():
    discount = 0.99
    reward = -0.04
    runMDP(P, reward, discount)

    discount = 0.5
    reward = -0.04
    runMDP(P, reward, discount)

    discount = 0.5
    reward = -0.25
    runMDP(P, reward, discount)
    
    
if __name__ == '__main__':
    main()
    print('\nExiting normally. Thank you')