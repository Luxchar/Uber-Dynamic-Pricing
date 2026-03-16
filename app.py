"""
🎯 Application Streamlit de Démonstration
Dynamic Pricing avec Deep Reinforcement Learning

Cette application permet de:
- Charger le meilleur modèle RL entraîné
- Lancer des simulations interactives
- Comparer RL vs baselines
- Visualiser les décisions de pricing en temps réel
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import pickle
import sys

# Imports du projet
sys.path.append(str(Path(__file__).parent / "src"))
from src.utils.pricing_env import DynamicPricingEnv
from src.utils.evaluation import (
    evaluate_policy_common,
    FixedPriceBaseline,
    HeuristicBaseline
)

# Configuration de la page
st.set_page_config(
    page_title="Dynamic Pricing RL Demo",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style
sns.set_style('whitegrid')

# ==========================================
# FONCTIONS UTILITAIRES
# ==========================================

@st.cache_resource
def load_model_and_data():
    """Charge le meilleur modèle et les données"""
    PROJECT_ROOT = Path(__file__).parent
    
    # Chemins
    model_path = PROJECT_ROOT / "artifacts" / "models" / "best_model.zip"
    metadata_path = PROJECT_ROOT / "artifacts" / "models" / "best_model_metadata.json"
    data_path = PROJECT_ROOT / "data" / "raw" / "dynamic_pricing_1M.csv"
    encoders_path = PROJECT_ROOT / "data" / "processed" / "label_encoders.pkl"
    
    # Vérifications
    if not model_path.exists():
        st.error(f"❌ Modèle non trouvé: {model_path}")
        st.info("👉 Veuillez d'abord exécuter le notebook d'entraînement.")
        st.stop()
    
    if not data_path.exists():
        st.error(f"❌ Dataset non trouvé: {data_path}")
        st.stop()
    
    # Chargement du modèle
    try:
        from stable_baselines3 import PPO, DQN, A2C, SAC, TD3
        from sb3_contrib import TQC
        
        # Lire les métadonnées pour connaître le type de modèle
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            model_class = metadata.get('model_class', 'PPO')
        else:
            model_class = 'PPO'  # Défaut
        
        # Mapper le nom de classe
        model_classes = {
            'PPO': PPO,
            'DQN': DQN,
            'A2C': A2C,
            'SAC': SAC,
            'TD3': TD3,
            'TQC': TQC
        }
        
        ModelClass = model_classes.get(model_class, PPO)
        model = ModelClass.load(str(model_path))
        
        st.success(f"✅ Modèle chargé: {model_class}")
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle: {e}")
        st.stop()
    
    # Chargement des données
    df = pd.read_csv(data_path)
    
    # Preprocessing
    if encoders_path.exists():
        with open(encoders_path, 'rb') as f:
            label_encoders = pickle.load(f)
        
        for col, le in label_encoders.items():
            df[f'{col}_encoded'] = le.transform(df[col])
    else:
        # Encodage simple si encodeurs pas disponibles
        from sklearn.preprocessing import LabelEncoder
        categorical_cols = ['Location_Category', 'Customer_Loyalty_Status', 'Time_of_Booking', 'Vehicle_Type']
        for col in categorical_cols:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col])
    
    # Ratio
    df['Ratio_Riders_Drivers'] = df['Number_of_Riders'] / (df['Number_of_Drivers'] + 1)
    
    # Échantillonnage pour l'app (10k lignes suffisent)
    df_sample = df.sample(n=min(10000, len(df)), random_state=42).reset_index(drop=True)
    
    # Métadonnées
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    return model, df_sample, metadata


def create_env_from_df(df, episode_length=50):
    """Créer un environnement à partir du DataFrame"""
    price_multipliers = np.array([0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
    env = DynamicPricingEnv(
        data=df,
        price_multipliers=price_multipliers,
        episode_length=episode_length,
        demand_elasticity=3.0,
        random_state=42
    )
    return env


def simulate_episode(env, policy, policy_name="Model"):
    """Simule un épisode complet et retourne l'historique"""
    obs, info = env.reset()
    done = False
    history = []
    
    while not done:
        action = policy(obs) if callable(policy) else policy.predict(obs, deterministic=True)[0]
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        history.append({
            'step': info['episode_step'] - 1,
            'action': info['action'],
            'multiplier': info['multiplier'],
            'proposed_price': info['proposed_price'],
            'accepted': info['accepted'],
            'reward': reward,
            'cumulative_reward': info['episode_revenue']
        })
    
    return pd.DataFrame(history)


# ==========================================
# INTERFACE PRINCIPALE
# ==========================================

def main():
    # Header
    st.title("🎯 Dynamic Pricing avec Deep RL")
    st.markdown("### Démonstration Interactive")
    st.markdown("---")
    
    # Chargement
    with st.spinner("Chargement du modèle et des données..."):
        model, df, metadata = load_model_and_data()
    
    # Sidebar: Configuration
    st.sidebar.header("⚙️ Configuration")
    
    episode_length = st.sidebar.slider(
        "Longueur de l'épisode (nombre de décisions)",
        min_value=10,
        max_value=100,
        value=50,
        step=10
    )
    
    n_simulations = st.sidebar.slider(
        "Nombre de simulations pour benchmark",
        min_value=10,
        max_value=100,
        value=50,
        step=10
    )
    
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Informations du Modèle")
    
    if metadata:
        st.sidebar.markdown(f"**Nom:** {metadata.get('model_name', 'N/A')}")
        st.sidebar.markdown(f"**Famille:** {metadata.get('model_family', 'N/A')}")
        st.sidebar.markdown(f"**Reward Test:** {metadata.get('test_mean_reward', 0):.2f}")
        st.sidebar.markdown(f"**Revenue Test:** {metadata.get('test_mean_revenue', 0):.2f}")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🎮 Simulation", "📊 Benchmark", "📈 Analyse"])
    
    # ========== TAB 1: SIMULATION ==========
    with tab1:
        st.header("🎮 Simulation Interactive")
        st.markdown("Lancez une simulation et comparez le modèle RL avec les baselines.")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            if st.button("🚀 Lancer une Simulation", type="primary", use_container_width=True):
                env = create_env_from_df(df, episode_length)
                
                # Simulation RL
                with st.spinner("Simulation du modèle RL..."):
                    history_rl = simulate_episode(env, model, "RL Model")
                
                # Simulation Baselines
                env_baseline = create_env_from_df(df, episode_length)
                baseline_fixed = FixedPriceBaseline(env.price_multipliers)
                history_fixed = simulate_episode(env_baseline, baseline_fixed, "Fixed Price")
                
                env_baseline2 = create_env_from_df(df, episode_length)
                baseline_heuristic = HeuristicBaseline(env.price_multipliers)
                history_heuristic = simulate_episode(env_baseline2, baseline_heuristic, "Heuristic")
                
                # Sauvegarde dans session state
                st.session_state['history_rl'] = history_rl
                st.session_state['history_fixed'] = history_fixed
                st.session_state['history_heuristic'] = history_heuristic
                st.success("✅ Simulation terminée!")
        
        with col2:
            if 'history_rl' in st.session_state:
                history_rl = st.session_state['history_rl']
                history_fixed = st.session_state['history_fixed']
                history_heuristic = st.session_state['history_heuristic']
                
                # Métriques comparatives
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.metric(
                        label="🤖 RL Model",
                        value=f"${history_rl['cumulative_reward'].iloc[-1]:.0f}",
                        delta=f"{history_rl['accepted'].mean()*100:.0f}% accepté"
                    )
                
                with col_b:
                    st.metric(
                        label="📊 Prix Fixe",
                        value=f"${history_fixed['cumulative_reward'].iloc[-1]:.0f}",
                        delta=f"{history_fixed['accepted'].mean()*100:.0f}% accepté"
                    )
                
                with col_c:
                    st.metric(
                        label="🧮 Heuristique",
                        value=f"${history_heuristic['cumulative_reward'].iloc[-1]:.0f}",
                        delta=f"{history_heuristic['accepted'].mean()*100:.0f}% accepté"
                    )
        
        # Graphiques
        if 'history_rl' in st.session_state:
            st.markdown("---")
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Revenu cumulatif
            axes[0, 0].plot(history_rl['step'], history_rl['cumulative_reward'], label='RL Model', linewidth=2)
            axes[0, 0].plot(history_fixed['step'], history_fixed['cumulative_reward'], label='Prix Fixe', linewidth=2, linestyle='--')
            axes[0, 0].plot(history_heuristic['step'], history_heuristic['cumulative_reward'], label='Heuristique', linewidth=2, linestyle='-.')
            axes[0, 0].set_title('Revenu Cumulatif')
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Revenu ($)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Multiplicateurs utilisés
            axes[0, 1].plot(history_rl['step'], history_rl['multiplier'], label='RL Model', alpha=0.7)
            axes[0, 1].axhline(y=1.0, color='red', linestyle='--', label='Prix normal (1.0x)', alpha=0.5)
            axes[0, 1].fill_between(history_rl['step'], 0.8, 1.5, alpha=0.1)
            axes[0, 1].set_title('Multiplicateurs de Prix (RL Model)')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Multiplicateur')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Acceptations
            window = 5
            rl_acceptance = history_rl['accepted'].rolling(window=window, min_periods=1).mean()
            fixed_acceptance = history_fixed['accepted'].rolling(window=window, min_periods=1).mean()
            heuristic_acceptance = history_heuristic['accepted'].rolling(window=window, min_periods=1).mean()
            
            axes[1, 0].plot(history_rl['step'], rl_acceptance, label='RL Model', linewidth=2)
            axes[1, 0].plot(history_fixed['step'], fixed_acceptance, label='Prix Fixe', linewidth=2, linestyle='--')
            axes[1, 0].plot(history_heuristic['step'], heuristic_acceptance, label='Heuristique', linewidth=2, linestyle='-.')
            axes[1, 0].set_title(f'Taux d\'Acceptation (moyenne mobile {window})')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Taux d\'acceptation')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_ylim([0, 1])
            
            # Distribution des actions RL
            action_counts = history_rl['action'].value_counts().sort_index()
            axes[1, 1].bar(action_counts.index, action_counts.values, color='skyblue', edgecolor='black')
            axes[1, 1].set_title('Distribution des Actions (RL Model)')
            axes[1, 1].set_xlabel('Action (indice)')
            axes[1, 1].set_ylabel('Fréquence')
            axes[1, 1].grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            st.pyplot(fig)
    
    # ========== TAB 2: BENCHMARK ==========
    with tab2:
        st.header("📊 Benchmark Comparatif")
        st.markdown("Comparez les performances sur plusieurs épisodes.")
        
        if st.button("🔄 Lancer le Benchmark", type="primary"):
            env = create_env_from_df(df, episode_length)
            
            # Évaluation RL
            with st.spinner("Évaluation du modèle RL..."):
                results_rl = evaluate_policy_common(
                    env=env,
                    policy=lambda obs: model.predict(obs, deterministic=True)[0],
                    n_eval_episodes=n_simulations,
                    model_name="RL Model (Best)",
                    model_family="Reinforcement Learning",
                    split_name="demo",
                    verbose=False
                )
            
            # Évaluation Baselines
            baseline_fixed = FixedPriceBaseline(env.price_multipliers)
            with st.spinner("Évaluation Prix Fixe..."):
                env_fixed = create_env_from_df(df, episode_length)
                results_fixed = evaluate_policy_common(
                    env=env_fixed,
                    policy=baseline_fixed,
                    n_eval_episodes=n_simulations,
                    model_name="Fixed Price",
                    model_family="Baseline",
                    split_name="demo",
                    verbose=False
                )
            
            baseline_heuristic = HeuristicBaseline(env.price_multipliers)
            with st.spinner("Évaluation Heuristique..."):
                env_heuristic = create_env_from_df(df, episode_length)
                results_heuristic = evaluate_policy_common(
                    env=env_heuristic,
                    policy=baseline_heuristic,
                    n_eval_episodes=n_simulations,
                    model_name="Heuristic",
                    model_family="Baseline",
                    split_name="demo",
                    verbose=False
                )
            
            # Tableau comparatif
            comparison_df = pd.DataFrame([
                {
                    'Modèle': results_rl['model_name'],
                    'Reward Moyen': f"{results_rl['mean_reward']:.2f} ± {results_rl['std_reward']:.2f}",
                    'Revenue Moyen': f"${results_rl['mean_revenue']:.2f}",
                    'Taux Acceptation': f"{results_rl['mean_acceptance_rate']*100:.1f}%",
                    'Mult. Moyen': f"{results_rl['mean_avg_multiplier']:.2f}x",
                    'Temps (s)': f"{results_rl['inference_time_total']:.2f}"
                },
                {
                    'Modèle': results_fixed['model_name'],
                    'Reward Moyen': f"{results_fixed['mean_reward']:.2f} ± {results_fixed['std_reward']:.2f}",
                    'Revenue Moyen': f"${results_fixed['mean_revenue']:.2f}",
                    'Taux Acceptation': f"{results_fixed['mean_acceptance_rate']*100:.1f}%",
                    'Mult. Moyen': f"{results_fixed['mean_avg_multiplier']:.2f}x",
                    'Temps (s)': f"{results_fixed['inference_time_total']:.2f}"
                },
                {
                    'Modèle': results_heuristic['model_name'],
                    'Reward Moyen': f"{results_heuristic['mean_reward']:.2f} ± {results_heuristic['std_reward']:.2f}",
                    'Revenue Moyen': f"${results_heuristic['mean_revenue']:.2f}",
                    'Taux Acceptation': f"{results_heuristic['mean_acceptance_rate']*100:.1f}%",
                    'Mult. Moyen': f"{results_heuristic['mean_avg_multiplier']:.2f}x",
                    'Temps (s)': f"{results_heuristic['inference_time_total']:.2f}"
                }
            ])
            
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            # Graphique de comparaison
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            models = [results_rl['model_name'], results_fixed['model_name'], results_heuristic['model_name']]
            rewards = [results_rl['mean_reward'], results_fixed['mean_reward'], results_heuristic['mean_reward']]
            revenues = [results_rl['mean_revenue'], results_fixed['mean_revenue'], results_heuristic['mean_revenue']]
            acceptance = [results_rl['mean_acceptance_rate']*100, results_fixed['mean_acceptance_rate']*100, results_heuristic['mean_acceptance_rate']*100]
            
            axes[0].bar(models, rewards, color=['green', 'orange', 'blue'], edgecolor='black')
            axes[0].set_title('Reward Moyen')
            axes[0].set_ylabel('Reward')
            axes[0].grid(True, alpha=0.3, axis='y')
            axes[0].tick_params(axis='x', rotation=15)
            
            axes[1].bar(models, revenues, color=['green', 'orange', 'blue'], edgecolor='black')
            axes[1].set_title('Revenue Moyen')
            axes[1].set_ylabel('Revenue ($)')
            axes[1].grid(True, alpha=0.3, axis='y')
            axes[1].tick_params(axis='x', rotation=15)
            
            axes[2].bar(models, acceptance, color=['green', 'orange', 'blue'], edgecolor='black')
            axes[2].set_title('Taux d\'Acceptation')
            axes[2].set_ylabel('Taux (%)')
            axes[2].set_ylim([0, 100])
            axes[2].grid(True, alpha=0.3, axis='y')
            axes[2].tick_params(axis='x', rotation=15)
            
            plt.tight_layout()
            st.pyplot(fig)
    
    # ========== TAB 3: ANALYSE ==========
    with tab3:
        st.header("📈 Analyse du Dataset")
        
        st.subheader("Distribution des Variables")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(df['Historical_Cost_of_Ride'], bins=50, edgecolor='black', alpha=0.7, color='skyblue')
            ax.set_title('Distribution: Prix Historique')
            ax.set_xlabel('Prix ($)')
            ax.set_ylabel('Fréquence')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(df['Ratio_Riders_Drivers'], bins=50, edgecolor='black', alpha=0.7, color='coral')
            ax.set_title('Distribution: Ratio Riders/Drivers')
            ax.set_xlabel('Ratio')
            ax.set_ylabel('Fréquence')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        st.subheader("Statistiques Descriptives")
        st.dataframe(df.describe(), use_container_width=True)
        
        st.subheader("Échantillon de Données")
        st.dataframe(df.head(20), use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("🎯 **Dynamic Pricing avec Deep RL** | Projet Académique 2026")


if __name__ == "__main__":
    main()
