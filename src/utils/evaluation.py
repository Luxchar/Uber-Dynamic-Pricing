"""
Fonctions d'évaluation communes pour tous les modèles (RL et baselines)

Ce module garantit que tous les modèles sont évalués de manière homogène
avec les mêmes métriques et la même méthodologie.
"""

import numpy as np
import time
from typing import Dict, Any, Callable, Optional
from pathlib import Path
import torch
import torch.nn as nn


def evaluate_policy_common(
    env,
    policy: Callable,
    n_eval_episodes: int = 100,
    deterministic: bool = True,
    model_name: str = "Unknown",
    model_family: str = "Unknown",
    split_name: str = "test",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Fonction d'évaluation universelle pour tous les modèles
    
    Cette fonction est utilisée pour ALL les baselines ET tous les modèles RL
    afin de garantir une comparaison strictement équitable.
    
    Args:
        env: Environnement Gymnasium
        policy: Fonction/modèle de politique callable(observation) -> action
        n_eval_episodes: Nombre d'épisodes d'évaluation
        deterministic: Utiliser la politique déterministe (pas d'exploration)
        model_name: Nom du modèle évalué
        model_family: Famille (Baseline / Value-Based / Policy-Based / Deep RL Advanced)
        split_name: Nom du split évalué (train/val/test)
        verbose: Afficher les progressions
        
    Returns:
        Dict contenant toutes les métriques standardisées
    """
    if verbose:
        print(f"\n🔍 Évaluation de {model_name} sur {split_name} ({n_eval_episodes} épisodes)...")
    
    # Métriques à collecter
    episode_rewards = []
    episode_revenues = []
    episode_lengths = []
    episode_acceptance_rates = []
    episode_avg_multipliers = []
    episode_price_volatilities = []
    
    # Chronomètre pour temps d'inférence
    start_time = time.time()
    
    # Boucle d'évaluation
    for episode_idx in range(n_eval_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        episode_revenue = 0.0
        episode_accepted = 0
        episode_steps = 0
        multipliers_used = []
        
        while not done:
            # Prédiction de l'action
            action = policy(obs)
            
            # Step dans l'environnement
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Collecte des métriques
            episode_reward += reward
            episode_revenue += info.get('reward', reward)
            episode_accepted += int(info.get('accepted', False))
            episode_steps += 1
            multipliers_used.append(info.get('multiplier', 1.0))
        
        # Statistiques de l'épisode
        episode_rewards.append(episode_reward)
        episode_revenues.append(episode_revenue)
        episode_lengths.append(episode_steps)
        episode_acceptance_rates.append(episode_accepted / episode_steps if episode_steps > 0 else 0.0)
        episode_avg_multipliers.append(np.mean(multipliers_used))
        
        # Volatilité des prix (écart-type des multiplicateurs utilisés)
        episode_price_volatilities.append(np.std(multipliers_used))
        
        if verbose and (episode_idx + 1) % 20 == 0:
            print(f"  Épisode {episode_idx + 1}/{n_eval_episodes} - "
                  f"Reward moyen: {np.mean(episode_rewards):.2f}")
    
    inference_time = time.time() - start_time
    
    # Agrégation des métriques
    results = {
        # Identification
        'model_name': model_name,
        'model_family': model_family,
        'split_evaluated': split_name,
        'n_episodes': n_eval_episodes,
        
        # Reward
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'min_reward': float(np.min(episode_rewards)),
        'max_reward': float(np.max(episode_rewards)),
        'median_reward': float(np.median(episode_rewards)),
        
        # Revenue
        'mean_revenue': float(np.mean(episode_revenues)),
        'std_revenue': float(np.std(episode_revenues)),
        'total_revenue': float(np.sum(episode_revenues)),
        
        # Acceptance
        'mean_acceptance_rate': float(np.mean(episode_acceptance_rates)),
        'std_acceptance_rate': float(np.std(episode_acceptance_rates)),
        
        # Pricing behavior
        'mean_avg_multiplier': float(np.mean(episode_avg_multipliers)),
        'mean_price_volatility': float(np.mean(episode_price_volatilities)),
        
        # Episode stats
        'mean_episode_length': float(np.mean(episode_lengths)),
        'std_episode_length': float(np.std(episode_lengths)),
        
        # Performance
        'inference_time_total': inference_time,
        'inference_time_per_episode': inference_time / n_eval_episodes,
        'inference_fps': (np.sum(episode_lengths) / inference_time) if inference_time > 0 else 0.0,
    }
    
    if verbose:
        print(f"\n✅ Évaluation terminée pour {model_name}")
        print(f"  📊 Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  💰 Mean Revenue: {results['mean_revenue']:.2f}")
        print(f"  ✓ Acceptance Rate: {results['mean_acceptance_rate']*100:.1f}%")
        print(f"  📈 Avg Multiplier: {results['mean_avg_multiplier']:.3f}")
        print(f"  ⚡ Inference time: {results['inference_time_total']:.2f}s ({results['inference_fps']:.1f} fps)")
    
    return results


class BaselinePolicy:
    """Classe de base pour les politiques baseline"""
    
    def __init__(self, name: str):
        self.name = name
    
    def __call__(self, obs: np.ndarray) -> int:
        """Interface commune: observation -> action"""
        raise NotImplementedError


class FixedPriceBaseline(BaselinePolicy):
    """
    Baseline 1: Prix Fixe
    Toujours choisir le multiplicateur 1.0 (prix normal)
    """
    
    def __init__(self, price_multipliers: np.ndarray):
        super().__init__("Fixed Price (1.0x)")
        self.price_multipliers = price_multipliers
        self.action = int(np.where(price_multipliers == 1.0)[0][0])
    
    def __call__(self, obs: np.ndarray) -> int:
        return self.action


class HeuristicBaseline(BaselinePolicy):
    """
    Baseline 2: Heuristique basée sur ratio offre/demande
    
    Logique:
    - Ratio élevé (forte demande) → surge pricing
    - Ratio faible (faible demande) → discount pricing
    """
    
    def __init__(self, price_multipliers: np.ndarray):
        super().__init__("Heuristic (Ratio-based)")
        self.price_multipliers = price_multipliers
        self.n_actions = len(price_multipliers)
    
    def __call__(self, obs: np.ndarray) -> int:
        # obs[2] = Ratio_Riders_Drivers
        ratio = obs[2]
        
        # Mappage ratio -> action
        # Ratio faible (<1) → discount
        # Ratio élevé (>3) → surge
        if ratio < 0.8:
            # Forte offre, faible demande → discount agressif
            action = 0  # 0.8x
        elif ratio < 1.2:
            # Équilibre → léger discount
            action = 1  # 0.9x
        elif ratio < 2.0:
            # Légère tension → prix normal
            action = 2  # 1.0x
        elif ratio < 3.0:
            # Tension modérée → léger surge
            action = 3  # 1.1x
        elif ratio < 4.0:
            # Forte tension → surge modéré
            action = 4  # 1.2x
        else:
            # Très forte tension → surge fort
            action = 5  # 1.3x
        
        return min(action, self.n_actions - 1)


class GreedyRegressorBaseline(BaselinePolicy):
    """
    Baseline 3: Politique Gloutonne via Régression Supervisée
    
    On entraîne un régresseur à prédire le revenu attendu pour chaque action,
    puis on choisit gloutonnement l'action avec le plus haut revenu prédit.
    """
    
    def __init__(
        self,
        price_multipliers: np.ndarray,
        env,
        n_samples: int = 10000
    ):
        super().__init__("Greedy Regressor")
        self.price_multipliers = price_multipliers
        self.n_actions = len(price_multipliers)
        
        # Entraînement du régresseur
        self._train_regressor(env, n_samples)
    
    def _train_regressor(self, env, n_samples):
        """Entraîne un régresseur simple pour estimer le revenu par action"""
        from sklearn.ensemble import RandomForestRegressor
        
        X_train = []
        y_train = []
        
        # Collecte de données (state, action) -> reward
        for _ in range(n_samples):
            obs, _ = env.reset()
            action = np.random.randint(0, self.n_actions)
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Feature: concatenation de obs + one-hot action
            action_onehot = np.zeros(self.n_actions)
            action_onehot[action] = 1
            features = np.concatenate([obs, action_onehot])
            
            X_train.append(features)
            y_train.append(reward)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Entraînement
        self.regressor = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        self.regressor.fit(X_train, y_train)
        
        print(f"✅ Regressor entraîné sur {n_samples} échantillons")
    
    def __call__(self, obs: np.ndarray) -> int:
        """Choisit l'action avec le plus haut revenu prédit"""
        # Prédiction pour toutes les actions
        predictions = []
        for action in range(self.n_actions):
            action_onehot = np.zeros(self.n_actions)
            action_onehot[action] = 1
            features = np.concatenate([obs, action_onehot]).reshape(1, -1)
            pred = self.regressor.predict(features)[0]
            predictions.append(pred)
        
        # Action gloutonne
        return int(np.argmax(predictions))


class _RewardMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


class NeuralBaseline(BaselinePolicy):
    """
    Baseline 4: Politique neuronale (PyTorch)

    Modèle supervisé simple qui prédit le reward attendu pour (state, action),
    puis choisit l'action gloutonne. Inference sur GPU si disponible et activé.
    """

    def __init__(self, price_multipliers: np.ndarray, env, n_samples: int = 7000, device: str = "cpu"):
        super().__init__("Neural Baseline (PyTorch)")
        self.price_multipliers = price_multipliers
        self.n_actions = len(price_multipliers)
        requested_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        self.device = torch.device(requested_device)
        self.model = _RewardMLP(input_dim=11 + self.n_actions).to(self.device)
        self._train_model(env, n_samples)

    def _train_model(self, env, n_samples: int):
        x_samples = []
        y_samples = []

        for _ in range(n_samples):
            obs, _ = env.reset()
            action = np.random.randint(0, self.n_actions)
            _, reward, _, _, _ = env.step(action)

            action_onehot = np.zeros(self.n_actions, dtype=np.float32)
            action_onehot[action] = 1.0
            features = np.concatenate([obs.astype(np.float32), action_onehot])

            x_samples.append(features)
            y_samples.append([float(reward)])

        x_train = torch.tensor(np.array(x_samples), dtype=torch.float32, device=self.device)
        y_train = torch.tensor(np.array(y_samples), dtype=torch.float32, device=self.device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        batch_size = 256
        epochs = 8

        self.model.train()
        n = x_train.size(0)
        for _ in range(epochs):
            indices = torch.randperm(n, device=self.device)
            for start in range(0, n, batch_size):
                batch_idx = indices[start:start + batch_size]
                xb = x_train[batch_idx]
                yb = y_train[batch_idx]

                optimizer.zero_grad()
                preds = self.model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()

        self.model.eval()
        print(f"✅ Neural baseline entraînée sur {n_samples} échantillons (device={self.device.type})")

    def __call__(self, obs: np.ndarray) -> int:
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
        obs_repeated = obs_tensor.unsqueeze(0).repeat(self.n_actions, 1)
        action_onehot = torch.eye(self.n_actions, dtype=torch.float32, device=self.device)
        features = torch.cat([obs_repeated, action_onehot], dim=1)

        with torch.no_grad():
            preds = self.model(features).squeeze(-1)

        return int(torch.argmax(preds).item())


def create_baseline_policies(
    price_multipliers: np.ndarray,
    env=None,
    use_gpu_baseline: bool = False,
    baseline_device: str = "cpu",
    include_greedy_regressor: bool = True
):
    """
    Créer toutes les baselines
    
    Args:
        price_multipliers: Array des multiplicateurs
        env: Environnement (nécessaire pour baselines supervisées)
        use_gpu_baseline: Ajouter une baseline PyTorch pouvant utiliser le GPU
        baseline_device: Device PyTorch visé ("cpu" ou "cuda")
        include_greedy_regressor: Inclure la baseline RF (plus lente)
        
    Returns:
        Dict de baselines
    """
    baselines = {
        'fixed_price': FixedPriceBaseline(price_multipliers),
        'heuristic': HeuristicBaseline(price_multipliers)
    }
    
    if env is not None:
        if include_greedy_regressor:
            baselines['greedy_regressor'] = GreedyRegressorBaseline(price_multipliers, env, n_samples=5000)
        if use_gpu_baseline:
            baselines['neural_torch'] = NeuralBaseline(
                price_multipliers=price_multipliers,
                env=env,
                n_samples=7000,
                device=baseline_device
            )
    
    return baselines
