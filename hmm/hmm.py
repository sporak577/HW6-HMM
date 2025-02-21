import numpy as np

"""
again used chatgpt to help me understand the code and also to set it up
i did also consult the literature provided on github 
"""

class HiddenMarkovModel:
    """
    Class for Hidden Markov Model 
    """

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_p: np.ndarray, transition_p: np.ndarray, emission_p: np.ndarray):
        """

        Initialization of HMM object

        Args:
            observation_states (np.ndarray): observed states 
            hidden_states (np.ndarray): hidden states 
            prior_p (np.ndarray): prior probabities of hidden states 
            transition_p (np.ndarray): transition probabilites between hidden states
            emission_p (np.ndarray): emission probabilites from transition to hidden states 
        """             
        
        self.observation_states = observation_states
        self.observation_states_dict = {state: index for index, state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {index: state for index, state in enumerate(list(self.hidden_states))}
        
        self.prior_p= prior_p
        self.transition_p = transition_p
        self.emission_p = emission_p


    def forward(self, input_observation_states: np.ndarray) -> float:
        """
        TODO 

        This function runs the forward algorithm on an input sequence of observation states

        Args:
            input_observation_states (np.ndarray): observation sequence to run forward algorithm on 

        Returns:
            forward_probability (float): forward probability (likelihood) for the input observed sequence  
        """        

        num_obs = len(input_observation_states)
        #determines how many possible hidden states exist 
        num_states = len(self.hidden_states)

        #step1: initialize the forward prob matrix
        alpha = np.zeros((num_obs, num_states))

        #initialize base case (step1)
        first_obs_index = self.observation_states_dict[input_observation_states[0]]
        #prob of starting in each hidden state and emitting the first observation, which is the prob of starting in hidden state multiplied by the 
        #prob of emitting the first observation given that hidden state. 
        #for each state si, we compute the prob of starting in state si and observing o1
        #self.emission_p is the emission prob of observing o1 in state si
        for state in range(num_states):
            alpha[0, state] = self.prior_p[state] * self.emission_p[state, first_obs_index]
       
        # Step 2. Calculate probabilities
        #for each time step t=2 to T, and for each state sj, we compute the prob of observing the first t obs and ending in state sj
        #we start with a loop that runs until the last obs, each iteration calculates the prob of raching each hidden state at time t, considering
        #all possible transitions from the previous step t-1
        for t in range(1, num_obs):
            #this index is used to look up the emission prob
            obs_index = self.observation_states_dict[input_observation_states[t]]
            for j in range(num_states):
            #loops through each hidden state at time t
            #takes the previous prob of all states, multiplies by the transition prob, sums over all prev. states to give total prob of arriving at state j at time t
            #after transitioning to thate Sj multiply prob of emitting observation t in this state which is self.emission_p
            #alpha[t-1, :] prob of being in certain state previousy, times self.transition_p[:, j] transitioning to desired state * prob of observation given desired state
                alpha[t, j] = np.sum(alpha[t-1, :] * self.transition_p[:, j]) * self.emission_p[j, obs_index]



        # Step 3. Return final probability 
        forward_probability = np.sum(alpha[-1, :])
        return forward_probability
        


    def viterbi(self, decode_observation_states: np.ndarray) -> list:
        """
        TODO

        This function runs the viterbi algorithm on an input sequence of observation states

        Args:
            decode_observation_states (np.ndarray): observation state sequence to decode 

        Returns:
            best_hidden_state_sequence(list): most likely list of hidden states that generated the sequence observed states
        """        
        num_obs = len(decode_observation_states)
        num_states = len(self.hidden_states)

        #store probabilities of hidden state at each step 
        viterbi_table = np.zeros((len(decode_observation_states), len(self.hidden_states)))
        backpointer = np.zeros((num_obs, num_states), dtype=int)  #stores best previous state

        #initialize base case (time step 1)
        first_obs_index = self.observation_states_dict[decode_observation_states[0]]
        for state in range(num_states):
            viterbi_table[0, state] = self.prior_p[state] * self.emission_p[state, first_obs_index]
            backpointer[0, state] = 0 #no previous state at t=0
        
        #step 2: compute viterbi probs iteratively, starts at t=2, converts the current obs into a numerical index
        for t in range(1, num_obs):
            #iterates through each time step
            obs_index = self.observation_states_dict[decode_observation_states[t]]
            for j in range(num_states):
                #selects the column for the current state
                probs = viterbi_table[t-1, :] * self.transition_p[:, j]
                #np.argmax finds the index of the highest prob transition
                best_prev_state = np.argmax(probs)
                viterbi_table[t, j] = probs[best_prev_state] * self.emission_p[j, obs_index]
                backpointer[t, j] = best_prev_state
            
        # Step 3. Traceback to find the best sequence
        best_last_state = np.argmax(viterbi_table[-1, :])
        best_hidden_state_sequence = [best_last_state]

        for t in range(num_obs-1, 0, -1):
            best_last_state = backpointer[t, best_last_state]
            best_hidden_state_sequence.append(best_last_state)
        
        #reversing the sequence to match the observation order
        best_hidden_state_sequence.reverse()

        #convert indices to state names
        best_hidden_state_sequence = [self.hidden_states_dict[state] for state in best_hidden_state_sequence]

        return best_hidden_state_sequence

        