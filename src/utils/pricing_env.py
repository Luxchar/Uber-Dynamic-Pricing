"""
Environnement Gymnasium Custom pour le Dynamic Pricing

Cet environnement simule un problème de tarification dynamique où l'agent
doit choisir des multiplicateurs de prix pour maximiser le revenu.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Any


class DynamicPricingEnv(gym.Env):
    """
    Environnement de Dynamic Pricing compatible Gymnasium/Stable-Baselines3
    
    État (State):
        - Nombre de riders (demande)
        - Nombre de drivers (offre)
        - Ratio riders/drivers
        - Durée attendue
        - Ratings moyens
        - Nombre de courses passées
        - Localisation (encodée)
        - Statut de fidélité (encodé)
        - Moment de réservation (encodé)
        - Type de véhicule (encodé)
        - Prix de base
    
    Action:
        Multiplicateur de prix discret (0-7) correspondant à [0.8, 0.9, 1.0, ..., 1.5]
    
    Récompense:
        Revenu obtenu si la course est acceptée, 0 sinon
        L'acceptation est déterminée par une fonction de demande simulée
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        data: pd.DataFrame,
        price_multipliers: np.ndarray = np.array([0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]),
        episode_length: int = 50,
        demand_elasticity: float = 3.0,
        random_state: Optional[int] = None
    ):
        """
        Initialise l'environnement
        
        Args:
            data: DataFrame contenant les données de courses
            price_multipliers: Array des multiplicateurs de prix possibles
            episode_length: Nombre de steps par épisode
            demand_elasticity: Sensibilité de la demande au prix (>0)
            random_state: Seed pour reproductibilité
        """
        super().__init__()
        
        self.data = data.reset_index(drop=True)
        self.price_multipliers = price_multipliers
        self.episode_length = episode_length
        self.demand_elasticity = demand_elasticity
        self.rng = np.random.default_rng(random_state)
        
        # Configuration de l'espace d'observation
        # 11 features continues
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(11,),
            dtype=np.float32
        )
        
        # Configuration de l'espace d'action
        # Actions discrètes: indices des multiplicateurs
        self.action_space = spaces.Discrete(len(self.price_multipliers))
        
        # État interne
        self.current_step = 0
        self.current_index = 0
        self.current_state = None
        self.episode_revenue = 0.0
        self.episode_accepted = 0
        self.episode_history = []
        
        # Statistiques pour normalisation (optionnel)
        self._compute_data_stats()
        
    def _compute_data_stats(self):
        """Calcule les statistiques du dataset pour normalisation"""
        self.data_mean = self.data[[
            'Number_of_Riders', 'Number_of_Drivers', 'Ratio_Riders_Drivers',
            'Expected_Ride_Duration', 'Average_Ratings', 'Number_of_Past_Rides',
            'Location_Category_encoded', 'Customer_Loyalty_Status_encoded',
            'Time_of_Booking_encoded', 'Vehicle_Type_encoded',
            'Historical_Cost_of_Ride'
        ]].mean().values
        
        self.data_std = self.data[[
            'Number_of_Riders', 'Number_of_Drivers', 'Ratio_Riders_Drivers',
            'Expected_Ride_Duration', 'Average_Ratings', 'Number_of_Past_Rides',
            'Location_Category_encoded', 'Customer_Loyalty_Status_encoded',
            'Time_of_Booking_encoded', 'Vehicle_Type_encoded',
            'Historical_Cost_of_Ride'
        ]].std().values + 1e-8
        
    def _get_observation(self, row_index: int) -> np.ndarray:
        """
        Construit le vecteur d'observation à partir d'une ligne du dataset
        
        Args:
            row_index: Index de la ligne dans le dataset
            
        Returns:
            Vecteur d'observation normalisé
        """
        row = self.data.iloc[row_index]
        
        obs = np.array([
            row['Number_of_Riders'],
            row['Number_of_Drivers'],
            row['Ratio_Riders_Drivers'],
            row['Expected_Ride_Duration'],
            row['Average_Ratings'],
            row['Number_of_Past_Rides'],
            row['Location_Category_encoded'],
            row['Customer_Loyalty_Status_encoded'],
            row['Time_of_Booking_encoded'],
            row['Vehicle_Type_encoded'],
            row['Historical_Cost_of_Ride']
        ], dtype=np.float32)
        
        # Normalisation (optionnelle, peut améliorer l'apprentissage)
        # obs = (obs - self.data_mean) / self.data_std
        
        return obs
    
    def _simulate_demand(
        self,
        base_price: float,
        proposed_price: float,
        ratio_riders_drivers: float,
        loyalty_status: int
    ) -> bool:
        """
        Simule l'acceptation d'une course selon le prix proposé
        
        Logique:
        - Prix bas → forte probabilité d'acceptation
        - Prix élevé → faible probabilité d'acceptation
        - Forte demande (ratio élevé) → tolérance aux prix élevés
        - Client fidèle → tolérance accrue
        
        Args:
            base_price: Prix de référence historique
            proposed_price: Prix proposé par l'agent
            ratio_riders_drivers: Ratio offre/demande
            loyalty_status: Statut de fidélité (0=Regular, 1=Silver, 2=Gold)
            
        Returns:
            True si course acceptée, False sinon
        """
        # Prix relatif: négatif si discount, positif si surge
        relative_price = (proposed_price - base_price) / base_price
        
        # Logit de l'acceptation
        # Coefficients calibrés pour un comportement réaliste
        logit = (
            2.0                                    # Intercept (biais vers acceptation)
            - self.demand_elasticity * relative_price  # Élasticité au prix
            + 0.5 * np.log(ratio_riders_drivers + 0.1)  # Effet demande/offre
            + 0.3 * loyalty_status                  # Effet fidélité
        )
        
        # Probabilité d'acceptation via sigmoïde
        prob_accept = 1.0 / (1.0 + np.exp(-logit))
        
        # Échantillonnage de Bernoulli
        return self.rng.random() < prob_accept
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Réinitialise l'environnement pour un nouvel épisode
        
        Args:
            seed: Seed optionnelle pour reproductibilité
            options: Options supplémentaires
            
        Returns:
            observation: État initial
            info: Informations supplémentaires
        """
        super().reset(seed=seed)
        
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        # Réinitialisation des compteurs
        self.current_step = 0
        self.episode_revenue = 0.0
        self.episode_accepted = 0
        self.episode_history = []
        
        # Échantillonnage d'un point de départ aléatoire dans le dataset
        self.current_index = self.rng.integers(0, len(self.data))
        
        # Observation initiale
        self.current_state = self._get_observation(self.current_index)
        
        info = {
            'episode_step': self.current_step,
            'episode_revenue': self.episode_revenue,
            'episode_accepted': self.episode_accepted
        }
        
        return self.current_state, info
    
    def step(
        self,
        action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Exécute une action dans l'environnement
        
        Args:
            action: Indice du multiplicateur de prix à appliquer
            
        Returns:
            observation: Nouvel état
            reward: Récompense obtenue
            terminated: Fin de l'épisode (horizon atteint)
            truncated: Épisode tronqué (pas utilisé ici)
            info: Informations supplémentaires
        """
        # Récupération de la ligne actuelle
        row = self.data.iloc[self.current_index]
        base_price = row['Historical_Cost_of_Ride']
        
        # Application du multiplicateur de prix
        price_multiplier = self.price_multipliers[action]
        proposed_price = base_price * price_multiplier
        
        # Simulation de la demande
        accepted = self._simulate_demand(
            base_price=base_price,
            proposed_price=proposed_price,
            ratio_riders_drivers=row['Ratio_Riders_Drivers'],
            loyalty_status=row['Customer_Loyalty_Status_encoded']
        )
        
        # Calcul de la récompense
        reward = proposed_price if accepted else 0.0
        
        # Mise à jour des statistiques
        self.episode_revenue += reward
        self.episode_accepted += int(accepted)
        
        # Historique
        self.episode_history.append({
            'step': self.current_step,
            'action': action,
            'multiplier': price_multiplier,
            'base_price': base_price,
            'proposed_price': proposed_price,
            'accepted': accepted,
            'reward': reward
        })
        
        # Passage au step suivant
        self.current_step += 1
        
        # Vérification de la fin d'épisode
        terminated = self.current_step >= self.episode_length
        truncated = False
        
        # Nouvel état (échantillonnage suivant dans le dataset)
        if not terminated:
            self.current_index = (self.current_index + 1) % len(self.data)
            self.current_state = self._get_observation(self.current_index)
        
        # Informations
        info = {
            'episode_step': self.current_step,
            'episode_revenue': self.episode_revenue,
            'episode_accepted': self.episode_accepted,
            'acceptance_rate': self.episode_accepted / self.current_step if self.current_step > 0 else 0.0,
            'avg_revenue_per_step': self.episode_revenue / self.current_step if self.current_step > 0 else 0.0,
            'action': action,
            'multiplier': price_multiplier,
            'accepted': accepted,
            'proposed_price': proposed_price
        }
        
        return self.current_state, reward, terminated, truncated, info
    
    def render(self, mode='human'):
        """Affichage de l'état actuel (optionnel)"""
        if mode == 'human':
            print(f"\n=== Step {self.current_step}/{self.episode_length} ===")
            print(f"Revenue cumulé: ${self.episode_revenue:.2f}")
            print(f"Courses acceptées: {self.episode_accepted}/{self.current_step}")
            if self.current_step > 0:
                print(f"Taux d'acceptation: {100*self.episode_accepted/self.current_step:.1f}%")
    
    def close(self):
        """Nettoyage (rien à faire ici)"""
        pass
    
    def get_episode_history(self) -> list:
        """Retourne l'historique de l'épisode en cours"""
        return self.episode_history
