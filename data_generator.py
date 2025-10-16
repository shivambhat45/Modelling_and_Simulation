import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class DataGenerator:
    """
    Generate synthetic datasets for healthcare patient flow simulation
    Based on parameters from the research paper (Table 1 & 2)
    """
    
    @staticmethod
    def generate_paper_parameters():
        """
        Generate parameter sets based on the research paper
        Returns default and various scenario parameters
        """
        scenarios = [
            {
                'name': 'Low Traffic',
                'lambda1': 2.0,
                'lambda2': 1.5,
                'mu1': 5.0,
                'mu2': 4.0,
                'eta1': 0.3,
                'eta2': 0.2,
                'theta1': 0.7,
                'theta2': 0.8,
                'description': 'Low patient arrival rate, system well below capacity'
            },
            {
                'name': 'Medium Traffic',
                'lambda1': 4.0,
                'lambda2': 3.5,
                'mu1': 6.0,
                'mu2': 5.5,
                'eta1': 0.4,
                'eta2': 0.3,
                'theta1': 0.6,
                'theta2': 0.7,
                'description': 'Moderate patient arrival rate, system operating normally'
            },
            {
                'name': 'High Traffic',
                'lambda1': 5.5,
                'lambda2': 5.0,
                'mu1': 7.0,
                'mu2': 6.5,
                'eta1': 0.5,
                'eta2': 0.4,
                'theta1': 0.5,
                'theta2': 0.6,
                'description': 'High patient arrival rate, system near capacity'
            },
            {
                'name': 'Critical Load',
                'lambda1': 6.5,
                'lambda2': 6.0,
                'mu1': 7.5,
                'mu2': 7.0,
                'eta1': 0.6,
                'eta2': 0.5,
                'theta1': 0.4,
                'theta2': 0.5,
                'description': 'Very high patient arrival rate, system at critical capacity'
            },
            {
                'name': 'Paper Example',
                'lambda1': 3.0,
                'lambda2': 2.5,
                'mu1': 5.5,
                'mu2': 5.0,
                'eta1': 0.35,
                'eta2': 0.25,
                'theta1': 0.65,
                'theta2': 0.75,
                'description': 'Example parameters from the research paper'
            }
        ]
        
        return scenarios
    
    @staticmethod
    def generate_observation_sequences(n_sequences=5, min_length=10, max_length=30):
        """
        Generate synthetic observation sequences for HMM
        Sequences consist of 'A' (Admission) and 'D' (Discharge) events
        """
        sequences = []
        np.random.seed(42)
        
        for i in range(n_sequences):
            length = np.random.randint(min_length, max_length + 1)
            
            # Generate with realistic patterns (more admissions followed by discharges)
            sequence = []
            for _ in range(length):
                if len(sequence) == 0 or sequence[-1] == 'D':
                    # More likely to have admission after discharge
                    sequence.append('A' if np.random.random() < 0.7 else 'D')
                else:
                    # After admission, could be another admission or discharge
                    sequence.append('A' if np.random.random() < 0.4 else 'D')
            
            sequences.append({
                'id': i + 1,
                'sequence': ''.join(sequence),
                'length': length
            })
        
        return sequences
    
    @staticmethod
    def generate_patient_flow_data(lambda1, lambda2, mu1, mu2, n_patients=100):
        """
        Generate synthetic patient flow data with timestamps
        
        Returns DataFrame with patient arrival, service, and departure times
        """
        np.random.seed(42)
        
        patients = []
        current_time = datetime.now()
        
        for i in range(n_patients):
            # Determine if patient goes to HR or Nursing first
            goes_to_hr = np.random.random() < (lambda1 / (lambda1 + lambda2))
            
            # Generate inter-arrival time (exponential distribution)
            inter_arrival = np.random.exponential(1 / (lambda1 + lambda2))
            arrival_time = current_time + timedelta(hours=inter_arrival)
            
            if goes_to_hr:
                # Service time at HR
                service_time_hr = np.random.exponential(1 / mu1)
                departure_time_hr = arrival_time + timedelta(hours=service_time_hr)
                
                # May transfer to nursing
                transfers_to_nurse = np.random.random() < 0.3
                
                if transfers_to_nurse:
                    service_time_nurse = np.random.exponential(1 / mu2)
                    final_departure = departure_time_hr + timedelta(hours=service_time_nurse)
                    path = 'HR → Nurse'
                else:
                    final_departure = departure_time_hr
                    path = 'HR → Discharge'
            else:
                # Service time at Nursing
                service_time_nurse = np.random.exponential(1 / mu2)
                departure_time_nurse = arrival_time + timedelta(hours=service_time_nurse)
                
                # May transfer to HR
                transfers_to_hr = np.random.random() < 0.2
                
                if transfers_to_hr:
                    service_time_hr = np.random.exponential(1 / mu1)
                    final_departure = departure_time_nurse + timedelta(hours=service_time_hr)
                    path = 'Nurse → HR'
                else:
                    final_departure = departure_time_nurse
                    path = 'Nurse → Discharge'
            
            # Calculate total time in system
            total_time = (final_departure - arrival_time).total_seconds() / 3600  # hours
            
            patients.append({
                'patient_id': f'P{i+1:04d}',
                'arrival_time': arrival_time.isoformat(),
                'departure_time': final_departure.isoformat(),
                'path': path,
                'total_time_hours': round(total_time, 2)
            })
            
            current_time = arrival_time
        
        return patients
    
    @staticmethod
    def generate_hmm_training_data():
        """
        Generate example HMM parameters for healthcare system
        """
        # Transition matrix: P(next_state | current_state)
        # States: [HR, Nurse]
        transition_matrix = [
            [0.6, 0.4],  # From HR: 60% stay in HR, 40% to Nurse
            [0.3, 0.7]   # From Nurse: 30% to HR, 70% stay in Nurse
        ]
        
        # Emission matrix: P(observation | state)
        # Observations: [Admission 'A', Discharge 'D']
        emission_matrix = [
            [0.7, 0.3],  # HR: 70% admissions, 30% discharges
            [0.4, 0.6]   # Nurse: 40% admissions, 60% discharges
        ]
        
        # Initial distribution: P(starting state)
        initial_distribution = [0.6, 0.4]  # 60% start at HR, 40% at Nurse
        
        return {
            'transition_matrix': transition_matrix,
            'emission_matrix': emission_matrix,
            'initial_distribution': initial_distribution,
            'states': ['HR', 'Nurse'],
            'observations': ['A', 'D']
        }
    
    @staticmethod
    def generate_comparison_data():
        """
        Generate data for comparing different arrival rates (like Table 2 in paper)
        """
        arrival_rates = np.arange(1.0, 8.0, 0.5)
        comparison_data = []
        
        for lambda_rate in arrival_rates:
            # Fixed service rates
            mu1, mu2 = 6.0, 5.5
            lambda1 = lambda_rate
            lambda2 = lambda_rate * 0.9
            
            # Calculate basic metrics
            rho = (lambda1 + lambda2) / (mu1 + mu2)
            
            comparison_data.append({
                'arrival_rate': round(lambda_rate, 2),
                'lambda1': round(lambda1, 2),
                'lambda2': round(lambda2, 2),
                'traffic_intensity': round(rho, 4),
                'is_stable': rho < 1
            })
        
        return comparison_data
