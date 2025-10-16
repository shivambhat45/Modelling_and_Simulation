import numpy as np
from scipy import linalg
import logging

logger = logging.getLogger(__name__)

class MGOneQueueModel:
    """
    M/G/1 Queue Model for Patient Flow in Healthcare System
    Uses Matrix-Geometric Method for steady-state analysis
    """
    
    def __init__(self, lambda1, lambda2, mu1, mu2, eta1, eta2, theta1, theta2):
        """
        Initialize the M/G/1 Queue Model with parameters
        
        Parameters:
        -----------
        lambda1: Arrival rate to HR department
        lambda2: Arrival rate to Nursing unit
        mu1: Service rate at HR department
        mu2: Service rate at Nursing unit
        eta1: Transition probability from HR to Nursing
        eta2: Transition probability from Nursing to HR
        theta1: Departure probability from HR
        theta2: Departure probability from Nursing
        """
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.mu1 = mu1
        self.mu2 = mu2
        self.eta1 = eta1
        self.eta2 = eta2
        self.theta1 = theta1
        self.theta2 = theta2
        
        # Calculate stability condition
        self.rho = self._calculate_traffic_intensity()
        self.is_stable = self.rho < 1
        
        logger.info(f"Queue Model initialized with ρ={self.rho:.4f}, stable={self.is_stable}")
    
    def _calculate_traffic_intensity(self):
        """
        Calculate traffic intensity ρ
        System is stable if ρ < 1
        """
        numerator = (self.theta1 + self.eta2) * self.lambda1 + (self.theta2 + self.eta1) * self.lambda2
        denominator = ((self.theta1 + self.eta2) * (self.mu1 + self.theta2) + 
                      (self.theta2 + self.eta1) * (self.mu2 + self.theta1))
        
        if denominator == 0:
            return float('inf')
        
        return numerator / denominator
    
    def _construct_generator_matrices(self):
        """
        Construct the infinitesimal generator matrices A0, A1, A2, B
        for the M/G/1 queue model
        """
        # A0: Arrival matrix (transitions with arrivals)
        A0 = np.array([
            [self.lambda1, self.lambda2],
            [self.lambda2, self.lambda1]
        ])
        
        # A1: State transition matrix (transitions within same level)
        A1 = np.array([
            [-(self.lambda1 + self.lambda2 + self.mu1), self.mu2],
            [self.mu1, -(self.lambda1 + self.lambda2 + self.mu2)]
        ])
        
        # A2: Service completion matrix (transitions with departures)
        A2 = np.array([
            [self.mu1 * self.theta1, self.mu2 * self.eta2],
            [self.mu1 * self.eta1, self.mu2 * self.theta2]
        ])
        
        # B: Boundary condition matrix
        B = A1 + A2
        
        return A0, A1, A2, B
    
    def calculate_R_matrix(self, max_iterations=1000, tolerance=1e-8):
        """
        Calculate the rate matrix R using iterative method
        R satisfies: R = A0 + R * A1 + R^2 * A2
        """
        if not self.is_stable:
            logger.warning("System is unstable (ρ >= 1). R matrix may not converge.")
            return None
        
        A0, A1, A2, B = self._construct_generator_matrices()
        
        # Initialize R as zero matrix
        R = np.zeros_like(A0)
        
        for iteration in range(max_iterations):
            R_old = R.copy()
            
            # Iterative equation: R = A0 * inv(-A1 - R * A2)
            try:
                temp = A1 + R @ A2
                R_new = -A0 @ linalg.inv(temp)
                R = R_new
            except linalg.LinAlgError:
                logger.error("Singular matrix encountered in R calculation")
                return None
            
            # Check convergence
            if np.max(np.abs(R - R_old)) < tolerance:
                logger.info(f"R matrix converged after {iteration + 1} iterations")
                return R
        
        logger.warning(f"R matrix did not converge after {max_iterations} iterations")
        return R
    
    def calculate_stationary_distribution(self, R):
        """
        Calculate the stationary probability distribution π
        using the matrix-geometric method
        """
        if R is None:
            return None
        
        A0, A1, A2, B = self._construct_generator_matrices()
        
        try:
            # Solve π0 * (B + R * A0) = 0 and π0 * e = 1
            # where e is a column vector of ones
            Q0 = B + R @ A0
            
            # Augment with normalization constraint
            # We need to solve: π0 * Q0 = 0 and π0 * (I - R)^-1 * e = 1
            I = np.eye(len(R))
            e = np.ones((len(R), 1))
            
            # Calculate (I - R)^-1
            I_minus_R_inv = linalg.inv(I - R)
            
            # Normalization factor
            norm_vector = (I_minus_R_inv @ e).flatten()
            
            # Construct augmented system
            # [Q0.T | norm_vector.T] [π0.T] = [0...0 | 1]
            aug_matrix = np.vstack([Q0.T, norm_vector])
            aug_rhs = np.zeros(len(Q0) + 1)
            aug_rhs[-1] = 1
            
            # Solve using least squares (more stable)
            pi0, residuals, rank, s = linalg.lstsq(aug_matrix, aug_rhs)
            
            # Ensure non-negative probabilities
            pi0 = np.abs(pi0)
            pi0 = pi0 / np.sum(pi0)  # Renormalize
            
            return pi0, R
            
        except (linalg.LinAlgError, ValueError) as e:
            logger.error(f"Error calculating stationary distribution: {e}")
            return None, R
    
    def get_system_info(self):
        """
        Return system information dictionary
        """
        return {
            'parameters': {
                'lambda1': self.lambda1,
                'lambda2': self.lambda2,
                'mu1': self.mu1,
                'mu2': self.mu2,
                'eta1': self.eta1,
                'eta2': self.eta2,
                'theta1': self.theta1,
                'theta2': self.theta2
            },
            'traffic_intensity': self.rho,
            'is_stable': self.is_stable
        }
