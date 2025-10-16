import numpy as np
import logging

logger = logging.getLogger(__name__)

class HiddenMarkovModel:
    """
    Hidden Markov Model for predicting patient admission and discharge sequences
    Implements:
    - Viterbi Algorithm: Find most likely state sequence
    - Baum-Welch Algorithm: Estimate HMM parameters
    - Forward-Backward Algorithm: Calculate probabilities
    """
    
    def __init__(self, n_states=2, n_observations=2):
        """
        Initialize HMM with default parameters
        
        Parameters:
        -----------
        n_states: Number of hidden states (default 2: HR, Nurse)
        n_observations: Number of observable symbols (default 2: Admission 'A', Discharge 'D')
        """
        self.n_states = n_states
        self.n_observations = n_observations
        
        # State names
        self.states = ['HR', 'Nurse']
        self.observations_map = {'A': 0, 'D': 1}  # Admission, Discharge
        self.observations_inv_map = {0: 'A', 1: 'D'}
        
        # Initialize with uniform distributions
        self.transition_matrix = np.ones((n_states, n_states)) / n_states
        self.emission_matrix = np.ones((n_states, n_observations)) / n_observations
        self.initial_distribution = np.ones(n_states) / n_states
        
        logger.info(f"HMM initialized with {n_states} states and {n_observations} observations")
    
    def set_parameters(self, transition_matrix, emission_matrix, initial_distribution):
        """
        Manually set HMM parameters
        """
        self.transition_matrix = np.array(transition_matrix)
        self.emission_matrix = np.array(emission_matrix)
        self.initial_distribution = np.array(initial_distribution)
        
        # Validate shapes
        assert self.transition_matrix.shape == (self.n_states, self.n_states)
        assert self.emission_matrix.shape == (self.n_states, self.n_observations)
        assert self.initial_distribution.shape == (self.n_states,)
        
        logger.info("HMM parameters set manually")
    
    def _observation_to_index(self, observation_sequence):
        """
        Convert observation sequence from symbols to indices
        """
        return [self.observations_map[obs] for obs in observation_sequence]
    
    def viterbi(self, observation_sequence):
        """
        Viterbi Algorithm: Find the most likely state sequence
        
        Parameters:
        -----------
        observation_sequence: List of observations (e.g., ['A', 'D', 'A', 'D'])
        
        Returns:
        --------
        most_likely_states: List of most likely hidden states
        max_probability: Probability of the most likely path
        """
        try:
            # Convert observations to indices
            obs_indices = self._observation_to_index(observation_sequence)
            T = len(obs_indices)
            
            # Initialize Viterbi variables
            viterbi_prob = np.zeros((self.n_states, T))
            backpointer = np.zeros((self.n_states, T), dtype=int)
            
            # Initialization step (t=0)
            viterbi_prob[:, 0] = self.initial_distribution * self.emission_matrix[:, obs_indices[0]]
            
            # Recursion step (t=1 to T-1)
            for t in range(1, T):
                for s in range(self.n_states):
                    # Calculate probabilities for all previous states
                    trans_probs = viterbi_prob[:, t-1] * self.transition_matrix[:, s]
                    
                    # Find maximum
                    backpointer[s, t] = np.argmax(trans_probs)
                    viterbi_prob[s, t] = np.max(trans_probs) * self.emission_matrix[s, obs_indices[t]]
            
            # Termination step
            best_last_state = np.argmax(viterbi_prob[:, T-1])
            max_probability = viterbi_prob[best_last_state, T-1]
            
            # Backtrack to find the most likely state sequence
            most_likely_states = [0] * T
            most_likely_states[T-1] = best_last_state
            
            for t in range(T-2, -1, -1):
                most_likely_states[t] = backpointer[most_likely_states[t+1], t+1]
            
            # Convert state indices to state names
            state_names = [self.states[s] for s in most_likely_states]
            
            logger.info(f"Viterbi algorithm completed. Max probability: {max_probability}")
            
            return {
                'states': state_names,
                'state_indices': [int(s) for s in most_likely_states],
                'probability': float(max_probability),
                'observation_sequence': list(observation_sequence)
            }
            
        except Exception as e:
            logger.error(f"Error in Viterbi algorithm: {e}")
            return None
    
    def forward(self, observation_sequence):
        """
        Forward Algorithm: Calculate probability of observation sequence
        """
        obs_indices = self._observation_to_index(observation_sequence)
        T = len(obs_indices)
        
        # Initialize forward variables
        alpha = np.zeros((self.n_states, T))
        
        # Initialization (t=0)
        alpha[:, 0] = self.initial_distribution * self.emission_matrix[:, obs_indices[0]]
        
        # Recursion (t=1 to T-1)
        for t in range(1, T):
            for s in range(self.n_states):
                alpha[s, t] = np.sum(alpha[:, t-1] * self.transition_matrix[:, s]) * self.emission_matrix[s, obs_indices[t]]
        
        # Termination
        probability = np.sum(alpha[:, T-1])
        
        return alpha, probability
    
    def backward(self, observation_sequence):
        """
        Backward Algorithm: Calculate backward probabilities
        """
        obs_indices = self._observation_to_index(observation_sequence)
        T = len(obs_indices)
        
        # Initialize backward variables
        beta = np.zeros((self.n_states, T))
        
        # Initialization (t=T-1)
        beta[:, T-1] = 1
        
        # Recursion (t=T-2 to 0)
        for t in range(T-2, -1, -1):
            for s in range(self.n_states):
                beta[s, t] = np.sum(self.transition_matrix[s, :] * self.emission_matrix[:, obs_indices[t+1]] * beta[:, t+1])
        
        return beta
    
    def baum_welch(self, observation_sequence, max_iterations=100, tolerance=1e-6):
        """
        Baum-Welch Algorithm: Estimate HMM parameters
        
        Parameters:
        -----------
        observation_sequence: List of observations
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance
        
        Returns:
        --------
        Dictionary with trained parameters
        """
        try:
            obs_indices = self._observation_to_index(observation_sequence)
            T = len(obs_indices)
            
            prev_log_likelihood = float('-inf')
            
            for iteration in range(max_iterations):
                # E-step: Forward-Backward
                alpha, forward_prob = self.forward(observation_sequence)
                beta = self.backward(observation_sequence)
                
                # Calculate xi (probability of being in state i at time t and state j at time t+1)
                xi = np.zeros((self.n_states, self.n_states, T-1))
                for t in range(T-1):
                    denominator = np.sum(alpha[:, t] * beta[:, t])
                    for i in range(self.n_states):
                        numerator = alpha[i, t] * self.transition_matrix[i, :] * self.emission_matrix[:, obs_indices[t+1]] * beta[:, t+1]
                        xi[i, :, t] = numerator / (denominator + 1e-10)
                
                # Calculate gamma (probability of being in state i at time t)
                gamma = np.sum(xi, axis=1)
                # Add the last time step
                gamma_last = alpha[:, T-1] * beta[:, T-1]
                gamma_last = gamma_last / (np.sum(gamma_last) + 1e-10)
                gamma = np.column_stack([gamma, gamma_last])
                
                # M-step: Update parameters
                # Update initial distribution
                self.initial_distribution = gamma[:, 0]
                
                # Update transition matrix
                for i in range(self.n_states):
                    for j in range(self.n_states):
                        self.transition_matrix[i, j] = np.sum(xi[i, j, :]) / (np.sum(gamma[i, :-1]) + 1e-10)
                
                # Update emission matrix
                for i in range(self.n_states):
                    for k in range(self.n_observations):
                        mask = (np.array(obs_indices) == k)
                        self.emission_matrix[i, k] = np.sum(gamma[i, mask]) / (np.sum(gamma[i, :]) + 1e-10)
                
                # Check convergence
                log_likelihood = np.log(forward_prob + 1e-10)
                
                if abs(log_likelihood - prev_log_likelihood) < tolerance:
                    logger.info(f"Baum-Welch converged after {iteration + 1} iterations")
                    break
                
                prev_log_likelihood = log_likelihood
            
            return {
                'transition_matrix': self.transition_matrix.tolist(),
                'emission_matrix': self.emission_matrix.tolist(),
                'initial_distribution': self.initial_distribution.tolist(),
                'iterations': iteration + 1,
                'log_likelihood': float(log_likelihood)
            }
            
        except Exception as e:
            logger.error(f"Error in Baum-Welch algorithm: {e}")
            return None
    
    def predict_next_observation(self, observation_sequence, n_steps=5):
        """
        Predict next n observations based on current sequence
        """
        # Use Viterbi to find most likely current state
        viterbi_result = self.viterbi(observation_sequence)
        if viterbi_result is None:
            return None
        
        current_state = viterbi_result['state_indices'][-1]
        
        predictions = []
        for _ in range(n_steps):
            # Predict next state
            next_state_probs = self.transition_matrix[current_state, :]
            next_state = np.argmax(next_state_probs)
            
            # Predict observation from next state
            obs_probs = self.emission_matrix[next_state, :]
            next_obs_idx = np.argmax(obs_probs)
            next_obs = self.observations_inv_map[next_obs_idx]
            
            predictions.append({
                'step': len(predictions) + 1,
                'predicted_state': self.states[next_state],
                'predicted_observation': next_obs,
                'confidence': float(np.max(obs_probs))
            })
            
            current_state = next_state
        
        return predictions
