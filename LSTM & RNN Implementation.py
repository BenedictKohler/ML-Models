# Implementation of a LSTM and Recurrent Neural Network

import numpy as np

def softmax(vector) :
    """ Returns new vector that sums to 1, whereby
    each output is the predicted probability that it is
    the desired output """
    top = np.exp(vector - np.max(vector)) # We subtract off the max in the matrice as e^x becomes large too quickly
    return top / top.sum(axis = 0) # divide by sum of each column


def sigmoid(matrice) :
    """ Sigmoid returns values between 0 and 1 which 
    usually correspond to predictions, but are used 
    in LSTM networks to 'forget' previous irrelevant information 
    by driving the value down to 0 """
    return 1 / (1 + np.exp(-matrice))

def tanh(matrice) :
    """
    tanh is the activation function in an LSTM
    """
    numerator = np.exp(matrice) - np.exp(-matrice)
    denominator = np.exp(matrice) + np.exp(-matrice)
    return numerator / denominator

# Recurrent neural network for a single timestep. The full network just repeats this
def rnn_cell_forward_prop(inputT, hidden_prev, parameters) :
    """
    inputT: input data at timestep t
    hidden_prev: previous hidden state / output at previous timestep
    parameters: Dictionary containing
                Wax -- Weight matrix multiplying the input
                Waa -- Weight matrix multiplying the hidden state
                Wya -- Weight matrix relating the hidden-state to the output
                ba --  Bias
                by -- Bias relating the hidden-state to the output
    
    Returns: next hidden state, this timesteps prediction and cache (values needed for backpropagation)
    """
    
    W_input = parameters["Wax"]
    W_hidden_state = parameters["Waa"]
    W_hidden_state_to_output = parameters["Wya"]
    b_input = parameters["ba"]
    b_hidden_state_to_output = parameters["by"]
    
    next_hidden = np.tanh(np.dot(W_input, inputT) + np.dot(W_hidden_state, hidden_prev) + b_input)
    prediction = softmax(np.dot(W_hidden_state_to_output, next_hidden) + b_hidden_state_to_output)
    
    cache = (next_hidden, hidden_prev, inputT, parameters)
    
    return next_hidden, prediction, cache

# Full forward propagation using above function
def rnn_forward(inputData, initial_hidden_state, parameters) :
    
    caches = []
    
    num_units, batch_size, num_timesteps = inputData.shape
    output_length, hidden_state_length = parameters["Wya"].shape
    
    hidden_state = np.zeros((hidden_state_length, batch_size, num_timesteps)) # Initial initialization
    prediction = np.zeros((output_length, batch_size, num_timesteps))
    
    next_hidden = initial_hidden_state

    for timestep in range(num_timesteps) :
        next_hidden, predictionT, cache = rnn_cell_forward_prop(inputData[:,:,timestep], next_hidden, parameters)
        hidden_state[:,:,timestep] = next_hidden
        prediction[:,:,timestep] = predictionT
        caches.append(cache)
    
    caches = (caches, inputData)
    
    return hidden_state, prediction, caches

# Backpropagating the error for one timestep
def rnn_cell_backprop(grad_next_hidden, cache) :
    
    next_hidden, hidden_prev, inputT, parameters = cache
    
    W_input = parameters["Wax"]
    W_hidden_state = parameters["Waa"]
    W_hidden_state_to_output = parameters["Wya"]
    b_input = parameters["ba"]
    b_hidden_state_to_output = parameters["by"]
    
    # Derivative of tanh(x) is 1-tanh(x)^2. In this case tanh(x) is next_hidden and grad_next_hidden
    # is inner function you mulitply by according to chain rule
    d_tanh = (1 - next_hidden ** 2) * grad_next_hidden
    
    # Calculate partial derivative with respect to input weights
    d_inputT = np.dot(W_input.T, d_tanh)
    d_W_input = np.dot(d_tanh, inputT.T);
    
    # Calculate partial derivative with respect to hidden states
    d_prev_hidden = np.dot(W_hidden_state.T, d_tanh)
    d_W_prev_hidden = np.dot(d_tanh, hidden_prev.T)
    
    # Calculate partial derivative with respect to bias
    d_bias_hidden = np.sum(d_tanh, axis = 1, keedims=True)
    
    # Cache results for updates
    gradients = {"dxt": d_inputT, "da_prev": d_prev_hidden, "dWax": d_W_input, 
                 "dWaa": d_W_prev_hidden, "dba": d_bias_hidden}
    
    return gradients
    
# Backpropagating error for all timesteps
def rnn_backpropagation(grads_hidden, caches) : # grads_hidden is all gradients of hidden states
    caches, inputData = caches
    
    hidden_1, initialHidden, input_1, parameters = caches[0]
    
    num_hidden_units, batch_size, num_timesteps = grads_hidden.shape
    num_input_units, batch_size = input_1.shape
    
    grad_input_data = np.zeros((num_input_units, batch_size, num_timesteps))
    grad_W_input = np.zeros((num_hidden_units, num_input_units))
    grad_W_hidden_state = np.zeros((num_hidden_units, num_hidden_units))
    grad_bias = np.zeros((num_hidden_units, 1))
    grad_initial_hidden = np.zeros((num_hidden_units, batch_size))
    grad_prev_hidden = np.zeros((num_hidden_units, batch_size))
    
    # start at last timestep and then compute erros back to start
    for timestep in reversed(range(num_timesteps)) :
        
        gradients = rnn_cell_backprop(grads_hidden[:, :, timestep] + grad_prev_hidden, caches[timestep])
        
        grad_input_t, grad_hidden_prev_t, grad_W_input_t, grad_W_hidden_state_t, grad_bias_t = gradients["dxt"], 
        gradients["da_prev"], gradients["dWax"], gradients["dWaa"], gradients["dba"]
        
        grad_input_data[:, :, timestep] = grad_input_t
        grad_W_input += grad_W_input_t
        grad_W_hidden_state += grad_W_hidden_state_t
        grad_bias += grad_bias_t
    
    grad_initial_hidden = grad_hidden_prev_t
    
    gradients = {"dx": grad_input_data, "da0": grad_initial_hidden, "dWax": grad_W_input, 
                 "dWaa": grad_W_hidden_state,"dba": grad_bias}
    
    return gradients


def lstm_cell_forward_prop(inputData, previous_hidden, previous_memory, parameters) :
    """
    parameters: Dictionary containing
                Wf -- Weight matrix of the forget gate
                bf -- Bias of the forget gate
                Wu -- Weight matrix of the update gate
                bi -- Bias of the update gate
                Wc -- Weight matrix of the first "tanh"
                bc -- Bias of the first "tanh"
                Wo -- Weight matrix of the output gate
                bo -- Bias of the output gate
                Wy -- Weight matrix relating the hidden-state to the output
                by -- Bias relating the hidden-state to the output
    
    Returns: 
        next_hidden: next hidden state
        next_memory: next memory state
        predictionT: prediction at timestep t
        cache: values needed for backpropagation
    """
    
    # Retrieve previous values
    Wf = parameters["Wf"]
    bf = parameters["bf"]
    Wu = parameters["Wu"]
    bu = parameters["bu"]
    Wm = parameters["Wm"]
    bm = parameters["bm"]
    Wo = parameters["Wo"]
    bo = parameters["bo"]
    W_hidden_to_out = parameters["Wy"]
    b_hidden_to_out = parameters["by"]
    
    # Retrive dimensions
    num_input_units, batch_size = inputData.shape
    output_length, hidden_state_length = W_hidden_to_out.shape
    
    # This combines previous hidden state and current input data
    input_hidden = np.zeros((hidden_state_length + num_input_units, batch_size))
    input_hidden[: hidden_state_length, :] = previous_hidden
    input_hidden[hidden_state_length :, :] = inputData

    # Update hidden state, memory, and produce output to determine error at this timestep
    forget_previous = sigmoid(np.dot(Wf, input_hidden) + bf)
    update_previous = sigmoid(np.dot(Wu, input_hidden) + bu)
    
    new_memory = tanh(np.dot(Wm, input_hidden) + bm)
    next_memory = forget_previous * previous_memory + update_previous * new_memory # Combine old memory states with new
    
    output_this_timestep = sigmoid(np.dot(Wo, input_hidden) + bo)
    
    next_hidden = output_this_timestep * tanh(next_memory)
    
    predictionT = softmax(np.dot(W_hidden_to_out, previous_hidden) + b_hidden_to_out)
    
    cache = (next_hidden, next_memory, previous_hidden, previous_memory, forget_previous, update_previous,
             new_memory, output_this_timestep, inputData, parameters)
    
    return next_hidden, next_memory, predictionT, cache
    
def lstm_forward(inputData, initialHidden, parameters) :
    
    caches = []
    
    # Retrieve initial dimensions
    num_units, batch_size, num_timesteps = inputData.shape
    num_outputs, hidden_state_size = parameters["Wy"].shape
    
    # Holds hidden, memory, and output states at each timestep. Used for backpropagation
    all_hidden_states = np.zeros((hidden_state_size, batch_size, num_timesteps))
    all_memory_states = np.zeros((hidden_state_size, batch_size, num_timesteps))
    predictions = np.zeros((num_outputs, batch_size, num_timesteps))
    
    # initialize hidden and memory
    next_hidden = initialHidden
    next_memory = np.zeros(next_hidden.shape) # hidden and memory have same dimensions
    
    for timestep in range(num_timesteps) :
        # Compute current timestep values
        next_hidden, next_memory, predictionT, cache = lstm_cell_forward_prop(inputData[:, :, timestep], next_hidden, 
                                                                              next_memory, parameters)
        
        # Add info of current timestep
        all_hidden_states[:, :, timestep] = next_hidden
        all_memory_states[:, :, timestep] = next_memory
        predictions[:, :, timestep] = predictionT
        
        caches.append(cache)
    
    caches = (caches, inputData)
    
    return all_hidden_states, predictions, all_memory_states, caches

def lstm_cell_backprop(d_next_hidden, d_next_memory, cache) :
    
    # Retrieve information from lstm forward step
    next_hidden, next_memory, previous_hidden, previous_memory, forget_previous, update_previous, \
    new_memory, output_this_timestep, inputData, parameters = cache
    
    # Retrieve dimensions
    num_input_units, batch_size = inputData.shape
    num_hidden_units, batch_size = next_hidden.shape
    
    # When backpropagating we calculate partial derivatives and then apply chain rule
    # This means we take derivatives with respect to each parameter while treating the others as constants
    # We then apply chain rule d/dx f(g(x)) = f'(g(x)) * g'(x) resulting in much multiplication
    # Note the derivative of tanh is (1 - tanh^2) and derivative of sigmoid is sigmoid * (1 - sigmoid)
    d_output_this_timestep = d_next_hidden * tanh(next_memory) * output_this_timestep * (1 - output_this_timestep)
    d_new_memory = (d_next_hidden * output_this_timestep * (1 - tanh(next_memory) ** 2) + d_next_memory) * output_this_timestep * (1 - new_memory ** 2)
    d_previous_update = (d_next_hidden * output_this_timestep * (1 - tanh(next_memory) ** 2) + d_next_memory) * new_memory * (1 - update_previous) * update_previous
    d_forget_previous = (d_next_hidden * output_this_timestep * (1 - tanh(next_memory) ** 2) + d_next_memory) * previous_memory * forget_previous * (1 - forget_previous)
    
    d_W_forget = np.dot(d_forget_previous, np.hstack([previous_hidden.T, inputData.T]))
    d_W_update = np.dot(d_previous_update, np.hstack([previous_hidden.T, inputData.T]))
    d_W_memory = np.dot(d_new_memory, np.hstack([previous_hidden.T, inputData.T]))
    d_W_output = np.dot(d_output_this_timestep, np.hstack([previous_hidden.T, inputData.T]))
    d_b_forget = np.sum(d_forget_previous, axis=1, keepdims=True)
    d_b_update = np.sum(d_previous_update, axis=1, keepdims=True)
    d_b_memory = np.sum(d_new_memory, axis=1, keepdims=True)
    d_b_output = np.sum(d_output_this_timestep, axis=1, keepdims=True)
    
    d_hidden_previous = np.dot(Wf[:, :num_hidden_units].T, d_forget_previous) + np.dot(Wm[:, :num_hidden_units].T, d_new_memory)
    + np.dot(Wu[:, :num_hidden_units].T, d_previous_update) + np.dot(W_hidden_to_out[:, :num_hidden_units].T, d_output_this_timestep)
    
    d_memory_previous = (d_next_hidden * output_this_timestep * (1 - tanh(next_memory) ** 2) + d_next_memory) * forget_previous
    
    d_inputData = np.dot(Wf[:, num_hidden_units:].T, d_forget_previous) + np.dot(Wm[:, num_hidden_units:].T, d_new_memory) 
    + np.dot(Wu[:, num_hidden_units:].T, d_previous_update) + np.dot(W_hidden_to_out[:, num_hidden_units:].T, d_output_this_timestep)
    
    gradients = {"dxt": d_inputData, "da_prev": d_forget_previous, "dc_prev": d_memory_previous, "dWf": d_W_forget,"dbf": d_b_forget,
                 "dWi": d_W_update,"dbi": d_b_update, "dWc": d_W_memory,"dbc": d_b_memory, "dWo": d_W_output,"dbo": d_b_output}
    
    return gradients

def lstm_backpropagation(d_hidden_states, caches) :
    
    caches, inputT = caches
    hidden_1, memory_1, initial_hidden, initial_memory, forget_1, update_1, new_memory_1, output_1, \
    inputT_1, parameters = caches[0]
    
    # Retrieve dimensions
    num_hidden_units, batch_size, num_timesteps = d_hidden_states.shape
    num_input_units, batch_size = inputT.shape
    
    # initialize the gradients
    d_inputData = np.zeros((num_input_units, batch_size, num_timesteps))
    d_initial_hidden = np.zeros((num_hidden_units, batch_size))
    d_hidden_prev = np.zeros((num_hidden_units, batch_size))
    d_memory_prev = np.zeros((num_hidden_units, batch_size))
    d_W_forget = np.zeros((num_hidden_units, num_hidden_units + num_input_units))
    d_W_update = np.zeros((num_hidden_units, num_hidden_units + num_input_units))
    d_W_memory = np.zeros((num_hidden_units, num_hidden_units + num_input_units))
    d_W_output = np.zeros((num_hidden_units, num_hidden_units + num_input_units))
    d_b_forget = np.zeros((num_hidden_units, 1))
    d_b_update = np.zeros((num_hidden_units, 1))
    d_b_memory = np.zeros((num_hidden_units, 1))
    d_b_output = np.zeros((num_hidden_units, 1))
    
    # Update for each timestep
    for timestep in reversed(range(num_timesteps)) :
        
        gradients = lstm_cell_backprop(d_hidden_states[:,:,timestep] + d_hidden_prev, d_memory_prev, caches[timestep])
        d_inputData[:,:,timestep] = gradients["dxt"]
        d_W_forget += gradients["dWf"]
        d_W_update += gradients["dWi"]
        d_W_memory += gradients["dWc"]
        d_W_output += gradients["dWo"]
        d_b_forget += gradients["dbf"]
        d_b_update += gradients["dbi"]
        d_b_memory += gradients["dbc"]
        d_b_output += gradients["dbo"]
        
    d_initial_hidden = gradients["da_prev"]
    
    gradients = {"dxt": d_inputData, "da0": d_initial_hidden, "dWf": d_W_forget,"dbf": d_b_forget,
                 "dWi": d_W_update,"dbi": d_b_update, "dWc": d_W_memory,"dbc": d_b_memory, "dWo": d_W_output,"dbo": d_b_output}

    return gradients





