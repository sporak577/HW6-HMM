import pytest
from hmm import HiddenMarkovModel
import numpy as np

"""
using chatgpt and the help of Isaiah Hazelwood (biophysics phd student)
"""

def test_mini_weather():
    """
    TODO: 
    Create an instance of your HMM class using the "small_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "small_weather_input_output.npz" file.

    Ensure that the output of your Forward algorithm is correct. 

    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    In addition, check for at least 2 edge cases using this toy model. 
    """

    mini_hmm=np.load('./data/mini_weather_hmm.npz')
    mini_input=np.load('./data/mini_weather_sequences.npz')

    #extract parameters from the .npz file 
    hidden_states = mini_hmm['hidden_states']
    observation_states = mini_hmm['observation_states']
    prior_p = mini_hmm['prior_p']
    transition_p = mini_hmm['transition_p']
    emission_p = mini_hmm['emission_p']

    #extract observation sequences and expected outputs
    observations = mini_input['observation_state_sequence']
    best_hidden_sequence = mini_input['best_hidden_state_sequence']

    # Check if observations are empty before running tests
    assert observations.size > 0

    print("Observations shape:", observations.shape)
    print("Observations content:", observations)
    
    #initialize the HMM model
    hmm = HiddenMarkovModel(observation_states, hidden_states, prior_p, transition_p, emission_p)

    forward_prob = hmm.forward(observations)

    viterbi_path = hmm.viterbi(observations)

    #converting numpy array to a list for proper comparison
    best_hidden_sequence = best_hidden_sequence.tolist()
    
    #there is only a 'best_hidden_state_sequence in the mini_input file, not for forward hmm
    assert viterbi_path == best_hidden_sequence    

    #edge cases
    empty_obs = np.array([])
    empty_obs = np.array([])
    assert empty_obs.size == 0, "Empty observation test case is not actually empty."
    assert hmm.forward(empty_obs) == 0.0 #forward should return 0 on empty sequence
    assert hmm.viterbi(empty_obs) == [] #"viterbi should return empty list for empty state

    single_obs = np.array([observation_states[0]])
    assert isinstance(hmm.forward(single_obs), float) #forward should return a probability, the total likelihood
    assert len(hmm.viterbi(single_obs)) == 1 #viterbi should return a single state, the most probable single state






def test_full_weather():

    """
    TODO: 
    Create an instance of your HMM class using the "full_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "full_weather_input_output.npz" file
        
    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    """

    pass

    full_hmm=np.load('./data/full_weather_hmm.npz')
    full_input=np.load('./data/full_weather_sequences.npz')

    #extract parameters from the .npz file 
    hidden_states = full_hmm['hidden_states']
    observation_states = full_hmm['observation_states']
    prior_p = full_hmm['prior_p']
    transition_p = full_hmm['transition_p']
    emission_p = full_hmm['emission_p']

    #extract observation sequences and expected outputs
    observations = full_input['observation_state_sequence']
    best_hidden_sequence = full_input['best_hidden_state_sequence']
    
    #initialize the HMM model
    hmm = HiddenMarkovModel(observation_states, hidden_states, prior_p, transition_p, emission_p)

    forward_prob = hmm.forward(observations)

    viterbi_path = hmm.viterbi(observations)

    #converting numpy array to a list for proper comparison
    best_hidden_sequence = best_hidden_sequence.tolist()
    
    #there is only a 'best_hidden_state_sequence in the mini_input file, not for forward hmm
    assert viterbi_path == best_hidden_sequence
    

    
   
    pass















