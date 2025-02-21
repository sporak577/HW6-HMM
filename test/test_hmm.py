import pytest
from hmm import HiddenMarkovModel
import numpy as np


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
    
    #initialize the HMM model
    hmm = HiddenMarkovModel(observation_states, hidden_states, prior_p, transition_p, emission_p)

    forward_prob = hmm.forward(observations)

    viterbi_path = hmm.viterbi(observations)

    #converting numpy array to a list for proper comparison
    best_hidden_sequence = best_hidden_sequence.tolist()
    
    #there is only a 'best_hidden_state_sequence in the mini_input file, not for forward hmm
    assert viterbi_path == best_hidden_sequence
    

    
   
    pass



def test_full_weather():

    """
    TODO: 
    Create an instance of your HMM class using the "full_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "full_weather_input_output.npz" file
        
    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    """

    pass













