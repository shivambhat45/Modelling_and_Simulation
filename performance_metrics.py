import numpy as np
from scipy import linalg
import logging

logger = logging.getLogger(__name__)

class PerformanceMetrics:
    """
    Calculate performance metrics for M/G/1 Queue Model
    - Expected number in system (Ln)
    - Expected number in queue (Lq)
    - Expected waiting time in system (Ws)
    - Expected waiting time in queue (Wq)
    """
    
    @staticmethod
    def calculate_metrics(queue_model, pi0, R):
        """
        Calculate all performance metrics
        
        Parameters:
        -----------
        queue_model: MGOneQueueModel instance
        pi0: Stationary probability distribution for boundary state
        R: Rate matrix from matrix-geometric method
        
        Returns:
        --------
        Dictionary containing all performance metrics
        """
        if pi0 is None or R is None:
            logger.error("Cannot calculate metrics: invalid pi0 or R matrix")
            return None
        
        try:
            I = np.eye(len(R))
            e = np.ones((len(R), 1))
            
            # Calculate (I - R)^-1
            I_minus_R_inv = linalg.inv(I - R)
            
            # Calculate (I - R)^-2
            I_minus_R_inv_squared = I_minus_R_inv @ I_minus_R_inv
            
            # Expected number of customers in the system (Ln)
            # Ln = π0 * (I - R)^-2 * e
            Ln = float(pi0 @ I_minus_R_inv_squared @ e)
            
            # Expected number of customers in the queue (Lq)
            # Construct A2 and B matrices
            A0, A1, A2, B = queue_model._construct_generator_matrices()
            B_inv = linalg.inv(B)
            
            # Lq = π0 * [(I - R)^-2 * e - A2 * B^-1 * e] - 1
            term1 = I_minus_R_inv_squared @ e
            term2 = A2 @ B_inv @ e
            Lq = float(pi0 @ (term1 - term2)) - 1
            
            # Ensure non-negative
            Lq = max(0, Lq)
            
            # Total arrival rate
            lambda_total = queue_model.lambda1 + queue_model.lambda2
            
            # Expected waiting time in system (Ws) - Little's Law
            Ws = Ln / lambda_total if lambda_total > 0 else 0
            
            # Expected waiting time in queue (Wq) - Little's Law
            Wq = Lq / lambda_total if lambda_total > 0 else 0
            
            # System utilization
            utilization = queue_model.rho
            
            # Average service rate
            avg_service_rate = (queue_model.mu1 + queue_model.mu2) / 2
            
            metrics = {
                'Ln': round(Ln, 4),  # Expected number in system
                'Lq': round(Lq, 4),  # Expected number in queue
                'Ws': round(Ws, 4),  # Expected waiting time in system
                'Wq': round(Wq, 4),  # Expected waiting time in queue
                'rho': round(utilization, 4),  # Traffic intensity
                'lambda_total': round(lambda_total, 4),  # Total arrival rate
                'avg_service_rate': round(avg_service_rate, 4),  # Average service rate
                'is_stable': queue_model.is_stable
            }
            
            logger.info(f"Performance metrics calculated: Ln={Ln:.4f}, Lq={Lq:.4f}, Ws={Ws:.4f}, Wq={Wq:.4f}")
            
            return metrics
            
        except (linalg.LinAlgError, ValueError, ZeroDivisionError) as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return None
    
    @staticmethod
    def calculate_time_dependent_metrics(queue_model, time_steps=100, max_time=10):
        """
        Simulate time-dependent queue behavior
        
        Parameters:
        -----------
        queue_model: MGOneQueueModel instance
        time_steps: Number of time steps to simulate
        max_time: Maximum simulation time
        
        Returns:
        --------
        Dictionary with time series data
        """
        times = np.linspace(0, max_time, time_steps)
        
        # Simplified simulation for visualization
        lambda_total = queue_model.lambda1 + queue_model.lambda2
        avg_service_rate = (queue_model.mu1 + queue_model.mu2) / 2
        
        # Generate sample paths using exponential inter-arrival and service times
        np.random.seed(42)  # For reproducibility
        
        queue_length_hr = []
        queue_length_nurse = []
        total_queue_length = []
        
        current_hr = 0
        current_nurse = 0
        
        for t in times:
            # Simple Markovian approximation
            # Probability of arrival
            if np.random.random() < lambda_total * (max_time / time_steps):
                if np.random.random() < queue_model.lambda1 / lambda_total:
                    current_hr += 1
                else:
                    current_nurse += 1
            
            # Probability of service completion
            if current_hr > 0 and np.random.random() < queue_model.mu1 * (max_time / time_steps):
                current_hr -= 1
                # Transition
                if np.random.random() < queue_model.eta1:
                    current_nurse += 1
            
            if current_nurse > 0 and np.random.random() < queue_model.mu2 * (max_time / time_steps):
                current_nurse -= 1
                # Transition
                if np.random.random() < queue_model.eta2:
                    current_hr += 1
            
            queue_length_hr.append(current_hr)
            queue_length_nurse.append(current_nurse)
            total_queue_length.append(current_hr + current_nurse)
        
        return {
            'times': times.tolist(),
            'queue_length_hr': queue_length_hr,
            'queue_length_nurse': queue_length_nurse,
            'total_queue_length': total_queue_length
        }
