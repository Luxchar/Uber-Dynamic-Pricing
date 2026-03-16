"""
Module d'analyse avancée pour les modèles RL
Ce module ajoute des analyses détaillées : courbes d'apprentissage, convergence, stabilité
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Any
from stable_baselines3.common.callbacks import BaseCallback


class DetailedLoggingCallback(BaseCallback):
    """
    Callback pour logger des métriques détaillées pendant l'entraînement
    """
    
    def __init__(self, eval_env, eval_freq: int = 1000, n_eval_episodes: int = 10, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        
        # Historiques
        self.timesteps = []
        self.mean_rewards = []
        self.std_rewards = []
        self.mean_revenues = []
        self.mean_acceptance_rates = []
        self.mean_episode_lengths = []
        
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Évaluation
            episode_rewards = []
            episode_revenues = []
            episode_acceptance_rates = []
            episode_lengths = []
            
            for _ in range(self.n_eval_episodes):
                obs, info = self.eval_env.reset()
                done = False
                episode_reward = 0
                episode_revenue = 0
                episode_accepted = 0
                episode_steps = 0
                
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    done = terminated or truncated
                    
                    episode_reward += reward
                    episode_revenue += reward
                    episode_accepted += int(info.get('accepted', False))
                    episode_steps += 1
                
                episode_rewards.append(episode_reward)
                episode_revenues.append(episode_revenue)
                episode_acceptance_rates.append(episode_accepted / episode_steps if episode_steps > 0 else 0)
                episode_lengths.append(episode_steps)
            
            # Enregistrement
            self.timesteps.append(self.num_timesteps)
            self.mean_rewards.append(np.mean(episode_rewards))
            self.std_rewards.append(np.std(episode_rewards))
            self.mean_revenues.append(np.mean(episode_revenues))
            self.mean_acceptance_rates.append(np.mean(episode_acceptance_rates))
            self.mean_episode_lengths.append(np.mean(episode_lengths))
            
            if self.verbose > 0:
                print(f"Timestep {self.num_timesteps}: Mean Reward = {self.mean_rewards[-1]:.2f} ± {self.std_rewards[-1]:.2f}")
        
        return True
    
    def get_history(self) -> Dict[str, List]:
        """Retourne l'historique complet"""
        return {
            'timesteps': self.timesteps,
            'mean_rewards': self.mean_rewards,
            'std_rewards': self.std_rewards,
            'mean_revenues': self.mean_revenues,
            'mean_acceptance_rates': self.mean_acceptance_rates,
            'mean_episode_lengths': self.mean_episode_lengths
        }


def plot_learning_curves(histories: Dict[str, Dict], save_path: Path = None):
    """
    Trace les courbes d'apprentissage pour tous les modèles
    
    Args:
        histories: Dict avec {model_name: history_dict}
        save_path: Chemin de sauvegarde (optionnel)
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for idx, (model_name, history) in enumerate(histories.items()):
        color = colors[idx % len(colors)]
        
        # Rewards
        axes[0, 0].plot(history['timesteps'], history['mean_rewards'], 
                       label=model_name, color=color, linewidth=2, alpha=0.8)
        axes[0, 0].fill_between(history['timesteps'],
                               np.array(history['mean_rewards']) - np.array(history['std_rewards']),
                               np.array(history['mean_rewards']) + np.array(history['std_rewards']),
                               color=color, alpha=0.2)
        
        # Revenue
        axes[0, 1].plot(history['timesteps'], history['mean_revenues'],
                       label=model_name, color=color, linewidth=2, alpha=0.8)
        
        # Acceptance rate
        axes[1, 0].plot(history['timesteps'], history['mean_acceptance_rates'],
                       label=model_name, color=color, linewidth=2, alpha=0.8)
        
        # Episode length
        axes[1, 1].plot(history['timesteps'], history['mean_episode_lengths'],
                       label=model_name, color=color, linewidth=2, alpha=0.8)
    
    # Configuration des graphiques
    axes[0, 0].set_title('Évolution du Reward Moyen', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Timesteps')
    axes[0, 0].set_ylabel('Reward Moyen')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('Évolution du Revenue Moyen', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Timesteps')
    axes[0, 1].set_ylabel('Revenue Moyen ($)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('Évolution du Taux d\'Acceptation', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Timesteps')
    axes[1, 0].set_ylabel('Taux d\'Acceptation')
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('Évolution de la Longueur d\'Épisode', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Timesteps')
    axes[1, 1].set_ylabel('Longueur Moyenne')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def analyze_convergence(history: Dict, model_name: str, window: int = 10, save_path: Path = None):
    """
    Analyse de convergence d'un modèle
    
    Args:
        history: Historique d'entraînement
        model_name: Nom du modèle
        window: Taille de la fenêtre pour moyenne mobile
        save_path: Chemin de sauvegarde
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    rewards = np.array(history['mean_rewards'])
    timesteps = np.array(history['timesteps'])
    
    # 1. Reward brut + moyenne mobile
    axes[0, 0].plot(timesteps, rewards, label='Reward', alpha=0.6, linewidth=1)
    if len(rewards) >= window:
        rolling_mean = pd.Series(rewards).rolling(window=window, min_periods=1).mean()
        axes[0, 0].plot(timesteps, rolling_mean, label=f'Moyenne mobile ({window})', 
                       color='red', linewidth=2)
    axes[0, 0].set_title(f'{model_name} - Évolution du Reward', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Timesteps')
    axes[0, 0].set_ylabel('Reward Moyen')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Stabilité (écart-type du reward)
    stds = np.array(history['std_rewards'])
    axes[0, 1].plot(timesteps, stds, color='orange', linewidth=2)
    axes[0, 1].fill_between(timesteps, 0, stds, color='orange', alpha=0.3)
    axes[0, 1].set_title(f'{model_name} - Stabilité (Écart-type)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Timesteps')
    axes[0, 1].set_ylabel('Écart-type du Reward')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Amélioration relative
    if len(rewards) > 1:
        improvements = np.diff(rewards)
        axes[1, 0].bar(timesteps[1:], improvements, color=['green' if x > 0 else 'red' for x in improvements],
                      alpha=0.6, width=(timesteps[1] - timesteps[0]) * 0.8)
        axes[1, 0].axhline(y=0, color='black', linestyle='--', linewidth=1)
        axes[1, 0].set_title(f'{model_name} - Amélioration par Évaluation', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Timesteps')
        axes[1, 0].set_ylabel('Δ Reward')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. Distribution des rewards (début vs fin)
    if len(rewards) >= 10:
        early_rewards = rewards[:len(rewards)//4]  # Premier quart
        late_rewards = rewards[-len(rewards)//4:]  # Dernier quart
        
        axes[1, 1].hist(early_rewards, bins=15, alpha=0.5, label='Début', color='blue', edgecolor='black')
        axes[1, 1].hist(late_rewards, bins=15, alpha=0.5, label='Fin', color='green', edgecolor='black')
        axes[1, 1].axvline(early_rewards.mean(), color='blue', linestyle='--', linewidth=2, label=f'Moy. Début: {early_rewards.mean():.2f}')
        axes[1, 1].axvline(late_rewards.mean(), color='green', linestyle='--', linewidth=2, label=f'Moy. Fin: {late_rewards.mean():.2f}')
        axes[1, 1].set_title(f'{model_name} - Distribution Rewards (Début vs Fin)', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Reward')
        axes[1, 1].set_ylabel('Fréquence')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    
    # Calcul de métriques de convergence
    if len(rewards) >= window:
        final_performance = rewards[-window:].mean()
        initial_performance = rewards[:window].mean()
        improvement = ((final_performance - initial_performance) / abs(initial_performance)) * 100
        
        final_stability = stds[-window:].mean()
        
        print(f"\n{'='*60}")
        print(f"ANALYSE DE CONVERGENCE: {model_name}")
        print(f"{'='*60}")
        print(f"Performance initiale: {initial_performance:.2f}")
        print(f"Performance finale: {final_performance:.2f}")
        print(f"Amélioration: {improvement:+.2f}%")
        print(f"Stabilité finale (std): {final_stability:.2f}")
        print(f"{'='*60}\n")


def compare_training_efficiency(histories: Dict[str, Dict], save_path: Path = None):
    """
    Compare l'efficacité d'apprentissage des différents modèles
    
    Args:
        histories: Dict avec {model_name: history_dict}
        save_path: Chemin de sauvegarde
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    # 1. Sample efficiency: reward vs timesteps
    for idx, (model_name, history) in enumerate(histories.items()):
        color = colors[idx % len(colors)]
        axes[0].plot(history['timesteps'], history['mean_rewards'],
                    label=model_name, color=color, linewidth=2, marker='o', markersize=4)
    
    axes[0].set_title('Sample Efficiency: Reward vs Timesteps', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Timesteps')
    axes[0].set_ylabel('Mean Reward')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Stabilité: coefficient de variation
    for idx, (model_name, history) in enumerate(histories.items()):
        color = colors[idx % len(colors)]
        cv = np.array(history['std_rewards']) / (np.array(history['mean_rewards']) + 1e-8)
        axes[1].plot(history['timesteps'], cv,
                    label=model_name, color=color, linewidth=2, marker='o', markersize=4)
    
    axes[1].set_title('Stabilité: Coefficient de Variation', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Timesteps')
    axes[1].set_ylabel('CV (std/mean)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def analyze_policy_behavior(env, model, model_name: str, n_episodes: int = 5, save_path: Path = None):
    """
    Analyse du comportement de la politique apprise
    
    Args:
        env: Environnement
        model: Modèle entraîné
        model_name: Nom du modèle
        n_episodes: Nombre d'épisodes à analyser
        save_path: Chemin de sauvegarde
    """
    all_states = []
    all_actions = []
    all_rewards = []
    all_multipliers = []
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            all_states.append(obs)
            all_actions.append(action)
            all_rewards.append(reward)
            all_multipliers.append(info.get('multiplier', 1.0))
            
            obs = next_obs
    
    all_states = np.array(all_states)
    all_actions = np.array(all_actions)
    all_rewards = np.array(all_rewards)
    all_multipliers = np.array(all_multipliers)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Distribution des actions
    unique, counts = np.unique(all_actions, return_counts=True)
    axes[0, 0].bar(unique, counts, color='skyblue', edgecolor='black')
    axes[0, 0].set_title(f'{model_name} - Distribution des Actions', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Action (indice)')
    axes[0, 0].set_ylabel('Fréquence')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # 2. Actions vs Ratio Riders/Drivers
    ratio_idx = 2  # Index du ratio dans l'état
    axes[0, 1].scatter(all_states[:, ratio_idx], all_actions, alpha=0.5, c=all_rewards, cmap='viridis')
    axes[0, 1].set_title(f'{model_name} - Actions vs Ratio Offre/Demande', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Ratio Riders/Drivers')
    axes[0, 1].set_ylabel('Action')
    axes[0, 1].colorbar(axes[0, 1].collections[0], label='Reward')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Multiplicateurs vs Base Price
    base_price_idx = -1  # Dernier élément de l'état
    axes[0, 2].scatter(all_states[:, base_price_idx], all_multipliers, alpha=0.5, c=all_rewards, cmap='viridis')
    axes[0, 2].set_title(f'{model_name} - Multiplicateurs vs Prix de Base', fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('Prix de Base ($)')
    axes[0, 2].set_ylabel('Multiplicateur')
    axes[0, 2].colorbar(axes[0, 2].collections[0], label='Reward')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Rewards vs Actions
    axes[1, 0].violinplot([all_rewards[all_actions == a] for a in unique if len(all_rewards[all_actions == a]) > 0],
                         positions=unique[[len(all_rewards[all_actions == a]) > 0 for a in unique]],
                         showmeans=True, showmedians=True)
    axes[1, 0].set_title(f'{model_name} - Distribution Rewards par Action', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Action')
    axes[1, 0].set_ylabel('Reward')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 5. Heatmap: Actions selon Riders et Drivers
    riders_idx = 0
    drivers_idx = 1
    riders_bins = np.linspace(all_states[:, riders_idx].min(), all_states[:, riders_idx].max(), 10)
    drivers_bins = np.linspace(all_states[:, drivers_idx].min(), all_states[:, drivers_idx].max(), 10)
    
    heatmap = np.zeros((len(riders_bins)-1, len(drivers_bins)-1))
    for i in range(len(riders_bins)-1):
        for j in range(len(drivers_bins)-1):
            mask = ((all_states[:, riders_idx] >= riders_bins[i]) & 
                   (all_states[:, riders_idx] < riders_bins[i+1]) &
                   (all_states[:, drivers_idx] >= drivers_bins[j]) & 
                   (all_states[:, drivers_idx] < drivers_bins[j+1]))
            if mask.sum() > 0:
                heatmap[i, j] = all_actions[mask].mean()
    
    im = axes[1, 1].imshow(heatmap, aspect='auto', cmap='RdYlGn', origin='lower')
    axes[1, 1].set_title(f'{model_name} - Action Moyenne (Riders vs Drivers)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Drivers (bins)')
    axes[1, 1].set_ylabel('Riders (bins)')
    plt.colorbar(im, ax=axes[1, 1], label='Action moyenne')
    
    # 6. Statistiques de la politique
    axes[1, 2].axis('off')
    stats_text = f"""
    STATISTIQUES DE LA POLITIQUE
    ════════════════════════════════
    
    Episodes analysés: {n_episodes}
    Décisions totales: {len(all_actions)}
    
    Action la plus fréquente: {unique[np.argmax(counts)]}
    Multiplicateur moyen: {all_multipliers.mean():.3f}
    Multiplicateur médian: {np.median(all_multipliers):.3f}
    
    Reward moyen: {all_rewards.mean():.2f}
    Reward médian: {np.median(all_rewards):.2f}
    Reward total: {all_rewards.sum():.2f}
    
    % Actions discount (<1.0): {(all_multipliers < 1.0).mean()*100:.1f}%
    % Actions normales (=1.0): {(all_multipliers == 1.0).mean()*100:.1f}%
    % Actions surge (>1.0): {(all_multipliers > 1.0).mean()*100:.1f}%
    """
    axes[1, 2].text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                   verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
