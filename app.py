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
import io
import zipfile
import sys
from typing import Dict, Any, Tuple

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

PROJECT_ROOT = Path(__file__).parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "dynamic_pricing_1M.csv"
ENCODERS_PATH = PROJECT_ROOT / "data" / "processed" / "label_encoders.pkl"
MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"
REPORTS_DIR = PROJECT_ROOT / "artifacts" / "reports"
PRICE_MULTIPLIERS = np.array([0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5])

CATEGORICAL_COLS = [
    'Location_Category',
    'Customer_Loyalty_Status',
    'Time_of_Booking',
    'Vehicle_Type'
]

FEATURE_COLUMNS = [
    'Number_of_Riders',
    'Number_of_Drivers',
    'Ratio_Riders_Drivers',
    'Expected_Ride_Duration',
    'Average_Ratings',
    'Number_of_Past_Rides',
    'Location_Category_encoded',
    'Customer_Loyalty_Status_encoded',
    'Time_of_Booking_encoded',
    'Vehicle_Type_encoded',
    'Historical_Cost_of_Ride'
]

MODEL_FILE_TO_NAME = {
    "dqn_final": "DQN",
    "a2c_final": "A2C",
    "ppo_final": "PPO",
    "dueling_dqn_final": "Dueling DQN",
    "recurrent_ppo_final": "Recurrent PPO",
    "sac_final": "SAC",
}

# ==========================================
# FONCTIONS UTILITAIRES
# ==========================================

@st.cache_data
def load_dataset_and_encoders() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Charge le dataset et prépare les colonnes nécessaires."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset non trouvé: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    label_encoders: Dict[str, Any] = {}

    if ENCODERS_PATH.exists():
        with open(ENCODERS_PATH, 'rb') as f:
            label_encoders = pickle.load(f)

    for col in CATEGORICAL_COLS:
        encoded_col = f"{col}_encoded"
        if encoded_col in df.columns:
            continue

        if col in label_encoders:
            df[encoded_col] = label_encoders[col].transform(df[col])
        else:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df[encoded_col] = le.fit_transform(df[col])
            label_encoders[col] = le

    if 'Ratio_Riders_Drivers' not in df.columns:
        df['Ratio_Riders_Drivers'] = df['Number_of_Riders'] / (df['Number_of_Drivers'] + 1)

    df_sample = df.sample(n=min(10000, len(df)), random_state=42).reset_index(drop=True)
    return df_sample, label_encoders


@st.cache_data
def load_benchmark() -> pd.DataFrame:
    benchmark_path = REPORTS_DIR / "benchmark_final.csv"
    if benchmark_path.exists():
        return pd.read_csv(benchmark_path)
    return pd.DataFrame()


@st.cache_data
def discover_models() -> Dict[str, Dict[str, Any]]:
    """Découvre tous les modèles RL disponibles dans artifacts/models."""
    registry: Dict[str, Dict[str, Any]] = {}

    metadata_best_path = MODELS_DIR / "best_model_metadata.json"
    best_metadata = {}
    if metadata_best_path.exists():
        with open(metadata_best_path, 'r') as f:
            best_metadata = json.load(f)

    for model_file in sorted(MODELS_DIR.glob("*.zip")):
        stem = model_file.stem
        if stem == "best_model":
            model_name = best_metadata.get("model_name", "Best Model")
            model_class = best_metadata.get("model_class", "PPO")
            display_name = f"{model_name} (best_model.zip)"
        else:
            model_name = MODEL_FILE_TO_NAME.get(stem, stem.replace("_", " ").title())
            if "recurrent" in stem:
                model_class = "RecurrentPPO"
            elif "dqn" in stem:
                model_class = "DQN"
            elif "a2c" in stem:
                model_class = "A2C"
            elif "ppo" in stem:
                model_class = "PPO"
            elif "sac" in stem:
                model_class = "SAC"
            else:
                model_class = "PPO"
            display_name = f"{model_name} ({model_file.name})"

        registry[display_name] = {
            "model_name": model_name,
            "class_name": model_class,
            "path": model_file,
            "file_name": model_file.name,
        }

    return registry


@st.cache_resource
def load_model(model_path: str, model_class_name: str):
    """Charge dynamiquement un modèle SB3/sb3-contrib."""
    from stable_baselines3 import PPO, DQN, A2C, SAC

    model_classes = {
        'PPO': PPO,
        'DQN': DQN,
        'A2C': A2C,
        'SAC': SAC,
    }

    try:
        from sb3_contrib import RecurrentPPO
        model_classes['RecurrentPPO'] = RecurrentPPO
    except Exception:
        pass

    ModelClass = model_classes.get(model_class_name, PPO)
    return ModelClass.load(model_path)


def select_best_model_by_metric(
    benchmark_df: pd.DataFrame,
    model_registry: Dict[str, Dict[str, Any]],
    metric: str
) -> str:
    """Sélectionne le meilleur modèle disponible selon la métrique choisie."""
    if not model_registry:
        return ""

    if benchmark_df.empty or metric not in benchmark_df.columns or "Model" not in benchmark_df.columns:
        return next(iter(model_registry.keys()))

    available_model_names = {meta["model_name"] for meta in model_registry.values()}
    candidates = benchmark_df[benchmark_df["Model"].isin(available_model_names)].copy()

    if candidates.empty:
        return next(iter(model_registry.keys()))

    ascending = metric == "Train_Time_s"
    candidates = candidates.sort_values(metric, ascending=ascending)
    best_name = candidates.iloc[0]["Model"]

    for display_name, meta in model_registry.items():
        if meta["model_name"] == best_name:
            return display_name

    return next(iter(model_registry.keys()))


def build_models_archive(model_registry: Dict[str, Dict[str, Any]]) -> bytes:
    """Construit une archive ZIP avec tous les modèles entraînés et fichiers utiles."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        for meta in model_registry.values():
            model_path = meta["path"]
            if model_path.exists():
                zf.write(model_path, arcname=f"models/{model_path.name}")

        benchmark_path = REPORTS_DIR / "benchmark_final.csv"
        if benchmark_path.exists():
            zf.write(benchmark_path, arcname="reports/benchmark_final.csv")

        metadata_path = MODELS_DIR / "best_model_metadata.json"
        if metadata_path.exists():
            zf.write(metadata_path, arcname="models/best_model_metadata.json")

    buffer.seek(0)
    return buffer.getvalue()


def create_env_from_df(df, episode_length=50):
    """Créer un environnement à partir du DataFrame"""
    env = DynamicPricingEnv(
        data=df,
        price_multipliers=PRICE_MULTIPLIERS,
        episode_length=episode_length,
        demand_elasticity=3.0,
        random_state=42
    )
    return env


def action_to_discrete_index(action, price_multipliers: np.ndarray) -> int:
    """Convertit une action potentiellement continue en index discret valide."""
    if np.isscalar(action):
        action_idx = int(action)
        return int(np.clip(action_idx, 0, len(price_multipliers) - 1))

    action_array = np.asarray(action).reshape(-1)
    action_value = float(action_array[0])
    nearest_idx = int(np.argmin(np.abs(price_multipliers - action_value)))
    return nearest_idx


def infer_action_details(action, price_multipliers: np.ndarray) -> Tuple[float, int, float]:
    """Retourne (action_brute, action_index, multiplicateur)."""
    if np.isscalar(action):
        action_idx = int(np.clip(int(action), 0, len(price_multipliers) - 1))
        multiplier = float(price_multipliers[action_idx])
        return float(action_idx), action_idx, multiplier

    action_array = np.asarray(action).reshape(-1)
    action_value = float(action_array[0])
    multiplier = float(np.clip(action_value, float(price_multipliers.min()), float(price_multipliers.max())))
    action_idx = int(np.argmin(np.abs(price_multipliers - multiplier)))
    return action_value, action_idx, multiplier


def simulate_episode(env, policy, policy_name="Model"):
    """Simule un épisode complet et retourne l'historique"""
    obs, info = env.reset()
    done = False
    history = []
    
    while not done:
        predicted_action = policy(obs) if callable(policy) else policy.predict(obs, deterministic=True)[0]
        action = action_to_discrete_index(predicted_action, env.price_multipliers)
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


def preprocess_input_dataframe(df_input: pd.DataFrame, label_encoders: Dict[str, Any], df_reference: pd.DataFrame) -> pd.DataFrame:
    """Prépare un DataFrame d'entrée (manuel ou CSV) pour la prédiction."""
    required_raw_columns = [
        'Number_of_Riders',
        'Number_of_Drivers',
        'Expected_Ride_Duration',
        'Average_Ratings',
        'Number_of_Past_Rides',
        'Historical_Cost_of_Ride',
        'Location_Category',
        'Customer_Loyalty_Status',
        'Time_of_Booking',
        'Vehicle_Type'
    ]

    missing = [col for col in required_raw_columns if col not in df_input.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}")

    df_processed = df_input.copy()

    for col in CATEGORICAL_COLS:
        encoded_col = f"{col}_encoded"
        if col in label_encoders:
            encoder = label_encoders[col]
            values = df_processed[col].astype(str)
            known_classes = set(map(str, encoder.classes_))
            unknown_values = sorted(set(values.unique()) - known_classes)
            if unknown_values:
                raise ValueError(f"Valeurs inconnues pour {col}: {unknown_values}")
            df_processed[encoded_col] = encoder.transform(values)
        else:
            reference_values = sorted(df_reference[col].astype(str).dropna().unique().tolist())
            mapping = {value: idx for idx, value in enumerate(reference_values)}
            values = df_processed[col].astype(str)
            unknown_values = sorted(set(values.unique()) - set(mapping.keys()))
            if unknown_values:
                raise ValueError(f"Valeurs inconnues pour {col}: {unknown_values}")
            df_processed[encoded_col] = values.map(mapping).astype(int)

    df_processed['Ratio_Riders_Drivers'] = (
        df_processed['Number_of_Riders'] / (df_processed['Number_of_Drivers'] + 1)
    )

    return df_processed


def predict_dataframe(model, df_processed: pd.DataFrame) -> pd.DataFrame:
    """Prédit les actions/multiplicateurs/prix pour chaque ligne d'entrée."""
    rows = []
    for idx, row in df_processed.iterrows():
        obs = row[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        action, _ = model.predict(obs, deterministic=True)
        raw_action, action_idx, multiplier = infer_action_details(action, PRICE_MULTIPLIERS)
        base_price = float(row['Historical_Cost_of_Ride'])
        proposed_price = base_price * multiplier

        rows.append({
            'row_id': idx,
            'predicted_action_raw': raw_action,
            'predicted_action_index': action_idx,
            'predicted_multiplier': multiplier,
            'base_price': base_price,
            'proposed_price': proposed_price
        })

    return pd.DataFrame(rows)


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
        try:
            df, label_encoders = load_dataset_and_encoders()
        except FileNotFoundError as e:
            st.error(f"❌ {e}")
            st.stop()

        model_registry = discover_models()
        benchmark_df = load_benchmark()

    if not model_registry:
        st.error(f"❌ Aucun modèle trouvé dans {MODELS_DIR}")
        st.info("👉 Exécute d'abord le notebook d'entraînement pour générer les fichiers .zip des modèles.")
        st.stop()
    
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
    st.sidebar.header("🏆 Sélection du Modèle")

    metric_options = ["Test_Reward", "Test_Revenue", "Test_Acceptance", "Train_Time_s"]
    metric_for_best = st.sidebar.selectbox(
        "Métrique pour sélectionner le meilleur modèle",
        options=metric_options,
        index=0
    )

    auto_best_model = select_best_model_by_metric(benchmark_df, model_registry, metric_for_best)
    model_selection_mode = st.sidebar.radio(
        "Mode de sélection",
        options=["Automatique (best metric)", "Manuel"],
        index=0
    )

    model_options = list(model_registry.keys())
    selected_model_display = auto_best_model
    if model_selection_mode == "Manuel":
        selected_model_display = st.sidebar.selectbox(
            "Choisir un modèle",
            options=model_options,
            index=model_options.index(auto_best_model) if auto_best_model in model_options else 0
        )

    selected_meta = model_registry[selected_model_display]

    with st.spinner(f"Chargement du modèle: {selected_meta['file_name']}..."):
        try:
            model = load_model(str(selected_meta["path"]), selected_meta["class_name"])
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement de {selected_meta['file_name']}: {e}")
            st.stop()

    st.sidebar.success(f"✅ Modèle actif: {selected_meta['model_name']}")
    
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Informations du Modèle")
    st.sidebar.markdown(f"**Nom:** {selected_meta['model_name']}")
    st.sidebar.markdown(f"**Classe:** {selected_meta['class_name']}")
    st.sidebar.markdown(f"**Fichier:** {selected_meta['file_name']}")

    if not benchmark_df.empty and "Model" in benchmark_df.columns:
        model_row = benchmark_df[benchmark_df["Model"] == selected_meta['model_name']]
        if not model_row.empty:
            model_row = model_row.iloc[0]
            st.sidebar.markdown(f"**Reward Test:** {model_row.get('Test_Reward', np.nan):.2f}")
            st.sidebar.markdown(f"**Revenue Test:** {model_row.get('Test_Revenue', np.nan):.2f}")
            st.sidebar.markdown(f"**Acceptance Test:** {model_row.get('Test_Acceptance', np.nan):.2f}")

    st.sidebar.markdown("---")
    archive_bytes = build_models_archive(model_registry)
    st.sidebar.download_button(
        label="📦 Télécharger tous les modèles (.zip)",
        data=archive_bytes,
        file_name="trained_models_bundle.zip",
        mime="application/zip",
        use_container_width=True
    )

    st.sidebar.caption(f"Modèles détectés: {len(model_registry)}")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎮 Simulation", "📊 Benchmark", "🔮 Prédiction", "📈 Analyse"])
    
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
                    history_rl = simulate_episode(env, model, selected_meta['model_name'])
                
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
                        label=f"🤖 {selected_meta['model_name']}",
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
                    policy=lambda obs: action_to_discrete_index(model.predict(obs, deterministic=True)[0], env.price_multipliers),
                    n_eval_episodes=n_simulations,
                    model_name=selected_meta['model_name'],
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

    # ========== TAB 3: PRÉDICTION ===========
    with tab3:
        st.header("🔮 Prédiction par Modèle RL")
        st.markdown("Prédisez les décisions de pricing depuis un formulaire manuel ou un CSV.")

        pred_mode = st.radio(
            "Mode d'entrée",
            options=["Saisie manuelle", "Upload CSV"],
            horizontal=True
        )

        if pred_mode == "Saisie manuelle":
            with st.form("manual_prediction_form"):
                st.subheader("Entrée manuelle")
                c1, c2, c3 = st.columns(3)

                with c1:
                    number_of_riders = st.number_input("Number_of_Riders", min_value=0.0, value=120.0, step=1.0)
                    number_of_drivers = st.number_input("Number_of_Drivers", min_value=0.0, value=80.0, step=1.0)
                    expected_duration = st.number_input("Expected_Ride_Duration", min_value=1.0, value=25.0, step=1.0)

                with c2:
                    avg_ratings = st.number_input("Average_Ratings", min_value=0.0, max_value=5.0, value=4.5, step=0.1)
                    n_past_rides = st.number_input("Number_of_Past_Rides", min_value=0.0, value=20.0, step=1.0)
                    base_price = st.number_input("Historical_Cost_of_Ride", min_value=1.0, value=45.0, step=1.0)

                with c3:
                    location = st.selectbox("Location_Category", sorted(df['Location_Category'].astype(str).unique().tolist()))
                    loyalty = st.selectbox("Customer_Loyalty_Status", sorted(df['Customer_Loyalty_Status'].astype(str).unique().tolist()))
                    booking_time = st.selectbox("Time_of_Booking", sorted(df['Time_of_Booking'].astype(str).unique().tolist()))
                    vehicle = st.selectbox("Vehicle_Type", sorted(df['Vehicle_Type'].astype(str).unique().tolist()))

                submit_manual = st.form_submit_button("Prédire", type="primary")

            if submit_manual:
                manual_df = pd.DataFrame([{
                    'Number_of_Riders': number_of_riders,
                    'Number_of_Drivers': number_of_drivers,
                    'Expected_Ride_Duration': expected_duration,
                    'Average_Ratings': avg_ratings,
                    'Number_of_Past_Rides': n_past_rides,
                    'Historical_Cost_of_Ride': base_price,
                    'Location_Category': location,
                    'Customer_Loyalty_Status': loyalty,
                    'Time_of_Booking': booking_time,
                    'Vehicle_Type': vehicle,
                }])

                try:
                    manual_processed = preprocess_input_dataframe(manual_df, label_encoders, df)
                    pred_df = predict_dataframe(model, manual_processed)
                    prediction = pred_df.iloc[0]

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Action prédite (index)", int(prediction['predicted_action_index']))
                    c2.metric("Multiplicateur", f"{prediction['predicted_multiplier']:.3f}x")
                    c3.metric("Prix proposé", f"${prediction['proposed_price']:.2f}")

                    st.dataframe(pred_df, use_container_width=True, hide_index=True)
                except ValueError as e:
                    st.error(f"❌ Entrée invalide: {e}")
                except Exception as e:
                    st.error(f"❌ Erreur de prédiction: {e}")

        else:
            st.subheader("Upload CSV")
            st.caption("Colonnes requises: Number_of_Riders, Number_of_Drivers, Expected_Ride_Duration, Average_Ratings, Number_of_Past_Rides, Historical_Cost_of_Ride, Location_Category, Customer_Loyalty_Status, Time_of_Booking, Vehicle_Type")

            uploaded_csv = st.file_uploader("Choisir un fichier CSV", type=["csv"])

            if uploaded_csv is not None:
                try:
                    input_df = pd.read_csv(uploaded_csv)
                    st.write("Aperçu des données importées:")
                    st.dataframe(input_df.head(10), use_container_width=True)

                    if st.button("Lancer les prédictions CSV", type="primary"):
                        processed_df = preprocess_input_dataframe(input_df, label_encoders, df)
                        pred_df = predict_dataframe(model, processed_df)
                        output_df = pd.concat([input_df.reset_index(drop=True), pred_df.drop(columns=['row_id'])], axis=1)

                        st.success(f"✅ Prédictions terminées: {len(output_df)} lignes")
                        st.dataframe(output_df.head(50), use_container_width=True)

                        output_csv = output_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="⬇️ Télécharger les prédictions CSV",
                            data=output_csv,
                            file_name="predictions_dynamic_pricing.csv",
                            mime="text/csv"
                        )
                except ValueError as e:
                    st.error(f"❌ Erreur de validation du CSV: {e}")
                except Exception as e:
                    st.error(f"❌ Erreur de lecture/prediction CSV: {e}")
    
    # ========== TAB 4: ANALYSE ==========
    with tab4:
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
