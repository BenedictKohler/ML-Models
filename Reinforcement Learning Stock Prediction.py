# Naive RL solution to stock price prediction
# Uses shallow neural network in order to learn the different
# states for optimal buy, sell, and hold trades.

import numpy as np
import math
import sys
import random
from collections import deque
from tensorflow.keras.layers import Conv1D, Dense, AveragePooling1D
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam

class Agent :
    
    def __init__(self, state_size) :
        self.state_size = state_size # size of all possible inputs
        self.action_size = 3 # sit, buy, sell
        self.memory = deque(maxlen=1000)
        self.inventory = []
        
        self.gamma = 0.95 
        self.epsilon = 1.0 # Determines probability of taking random action in order to gain experience
        self.epsilon_min = 0.01 # When model is trained, only take exploratory action 1% of the time
        self.epsilon_decay = 0.995 # As the model has increased practice we must stop it from choosing random actions as often
        
        self.model = self._model()
    
    def _model(self) :
        model = Sequential()
        model.add(Dense(units=64, input_dim=self.state_size, activation="relu"))
        model.add(Dense(units=32, activation="relu"))
        model.add(Dense(units=8, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=0.001))
        
        return model
    
    def act(self, state) :
        
        # This is exploration in order to find best actions given the state
        if random.random() <= self.epsilon :
            return random.randrange(self.action_size)
        
        # This is exploitation. Choose best action given the state
        options = self.model.predict(state)
        return np.argmax(options[0])
    
    def expReplay(self, batch_size) :
        mini_batch = []
        l_memory = len(self.memory)
        
        for i in range(l_memory - batch_size + 1, l_memory) :
            mini_batch.append(self.memory[i])
        
        for state, action, reward, next_state, done in mini_batch :
            target = reward
            if not done :
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min :
            self.epsilon *= self.epsilon_decay


def formatPrice(n) :
    return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

def getStockDataVec(key) :
    vec = []
    lines = open("data/" + key + ".csv", "r").read().splitlines()
    
    for line in lines[1:] :
        vec.append(float(line.split(",")[4]))
    
    return vec

def sigmoid(x) :
    return 1 / (1 + math.exp(-x))

# returns an an n-day state representation ending at time t
def getState(data, t, n) :
    d = t - n + 1
    block = data[d:t+1] if d >= 0 else -d * [data[0]] + data[0:t+1]
    res = []
    for i in range(n-1) :
        res.append(sigmoid(block[i+1] - block[i]))
    
    return np.array([res])


stock_name = sys.argv[1]
window_size = model.layers[0].input.shape.as_list()[1]

agent = Agent(window_size, True)
data = getStockDataVec(stock_name)
l = len(data) - 1
batch_size = 32

state = getState(data, 0, window_size + 1)
total_profit = 0
agent.inventory = []

for t in range(l):
	action = agent.act(state)

	# sit
	next_state = getState(data, t + 1, window_size + 1)
	reward = 0

	if action == 1: # buy
		agent.inventory.append(data[t])
		print("Buy: " + formatPrice(data[t]))

	elif action == 2 and len(agent.inventory) > 0: # sell
		bought_price = agent.inventory.pop(0)
		reward = max(data[t] - bought_price, 0)
		total_profit += data[t] - bought_price
		print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))

	done = True if t == l - 1 else False
	agent.memory.append((state, action, reward, next_state, done))
	state = next_state

	if done:
		print("--------------------------------")
		print(stock_name + " Total Profit: " + formatPrice(total_profit))
		print("--------------------------------")
        

stock_name, window_size, episode_count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

agent = Agent(window_size)
data = getStockDataVec(stock_name)
l = len(data) - 1
batch_size = 32

for e in range(episode_count + 1) :
    print("Episode " + str(e) + "/" + str(episode_count))
    state = getState(data, 0, window_size + 1)
    
    total_profit = 0
    agent.inventory = []
    
    for t in range(l) :
        action = agent.act(state)
        
        # Sit
        next_state = getState(data, t+1, window_size+1)
        reward = 0
        
        # Buy
        if action == 1 :
            agent.inventory.append(data[t])
            print("Buy: " + formatPrice(data[t]))
        
        # Sell
        elif action == 2 and len(agent.inventory) > 0 :
            bought_price = agent.inventory.pop(0)
			reward = max(data[t] - bought_price, 0)
			total_profit += data[t] - bought_price
			print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))

		done = True if t == l - 1 else False
		agent.memory.append((state, action, reward, next_state, done))
		state = next_state

		if done:
			print("--------------------------------")
			print("Total Profit: " + formatPrice(total_profit))
			print("--------------------------------")

		if len(agent.memory) > batch_size:
			agent.expReplay(batch_size)

	if e % 10 == 0:
		agent.model.save("models/model_ep" + str(e))
