# Modèles RL Supplémentaires pour le Notebook Enrichi
# À insérer après PPO et avant les analyses

# %% [markdown]
# # 5bis. 🆕 Algorithmes RL Avancés Supplémentaires
#
# Pour enrichir notre benchmark, nous ajoutons 3 modèles supplémentaires :
#
# 1. **Dueling DQN** : Architecture améliorée de DQN avec séparation value/advantage
# 2. **Recurrent PPO** : PPO avec LSTM pour capturer dépendances temporelles
# 3. **SAC (Soft Actor-Critic)** : Algorithme state-of-the-art pour actions continues
#
# **Note sur SAC :** Notre environnement est discret, mais nous pouvons adapter l'espace d'action
# en mode continu où l'agent choisit un multiplicateur réel dans [0.8, 1.5], puis on le discrétise.

# %%
# =============================================================================
# DUELING DQN avec Logging Détaillé
# =============================================================================
print("\\n" + "="*80)
print("ENTRAINEMENT: Dueling DQN (Architecture Améliorée)")
print(f"Device: {SB3_DEVICE}")
print("="*80)

from stable_baselines3.dqn import DQN as BaseDQN

logging_callback_dueling_dqn = DetailedLoggingCallback(
    eval_env=env_val,
    eval_freq=EVAL_FREQ,
    n_eval_episodes=N_EVAL_EPISODES,
    verbose=1
)

# Dueling DQN utilise une architecture avec séparation value/advantage
model_dueling_dqn = DQN(
    "MlpPolicy",
    env_train,
    learning_rate=1e-4,
    buffer_size=100_000,
    learning_starts=1000,
    batch_size=32,
    tau=0.005,
    gamma=0.99,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=1000,
    exploration_fraction=0.1,
    exploration_final_eps=0.05,
    policy_kwargs=dict(
        net_arch=[256, 256],  # Architecture plus profonde
        dueling=True  # Active l'architecture Dueling
    ),
    verbose=1,
    device=SB3_DEVICE,
    tensorboard_log=str(LOGS_DIR / "dueling_dqn")
)

print("\\n🚀 Début entraînement Dueling DQN...")
start_time = time.time()
model_dueling_dqn.learn(total_timesteps=TOTAL_TIMESTEPS, callback=logging_callback_dueling_dqn, progress_bar=True)
dueling_dqn_train_time = time.time() - start_time

history_dueling_dqn = logging_callback_dueling_dqn.get_history()
with open(LOGS_DIR / "dueling_dqn_history.json", 'w') as f:
    json.dump(history_dueling_dqn, f, indent=2)

model_dueling_dqn.save(MODELS_DIR / "dueling_dqn_final")

print(f"\\n✅ Dueling DQN entraîné en {dueling_dqn_train_time:.1f}s ({dueling_dqn_train_time/60:.1f} min)")

results_dueling_dqn_val = evaluate_policy_common(
    env=env_val,
    policy=lambda obs: model_dueling_dqn.predict(obs, deterministic=True)[0],
    n_eval_episodes=100,
    model_name="Dueling DQN",
    model_family="Value-Based Advanced",
    split_name="validation"
)

results_dueling_dqn_test = evaluate_policy_common(
    env=env_test,
    policy=lambda obs: model_dueling_dqn.predict(obs, deterministic=True)[0],
    n_eval_episodes=100,
    model_name="Dueling DQN",
    model_family="Value-Based Advanced",
    split_name="test"
)

# %% [markdown]
# ## 🧠 Dueling DQN - Architecture Améliorée
#
# **Principe :** Dueling DQN sépare l'estimation de la value function en deux composantes :
# - **Value Stream V(s)** : Valeur intrinsèque de l'état
# - **Advantage Stream A(s,a)** : Avantage de chaque action
#
# **Formule :** Q(s,a) = V(s) + (A(s,a) - mean(A(s,·)))
#
# **Avantages :**
# - Meilleure généralisation
# - Apprentissage plus stable
# - Converge souvent plus vite que DQN standard
#
# **Attentes :** Performance similaire ou supérieure à DQN avec meilleure stabilité

# %%
# =============================================================================
# RECURRENT PPO (avec LSTM) - Logging Détaillé
# =============================================================================
print("\\n" + "="*80)
print("ENTRAINEMENT: Recurrent PPO (avec LSTM)")
print(f"Device: {SB3_DEVICE}")
print("="*80)

from sb3_contrib import RecurrentPPO

logging_callback_recurrent_ppo = DetailedLoggingCallback(
    eval_env=env_val,
    eval_freq=EVAL_FREQ,
    n_eval_episodes=N_EVAL_EPISODES,
    verbose=1
)

# RecurrentPPO utilise un LSTM pour capturer les dépendances temporelles
model_recurrent_ppo = RecurrentPPO(
    "MlpLstmPolicy",
    env_train,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=dict(
        lstm_hidden_size=128,  # Taille du LSTM
        n_lstm_layers=1,
        enable_critic_lstm=True  # LSTM pour critic aussi
    ),
    verbose=1,
    device=SB3_DEVICE,
    tensorboard_log=str(LOGS_DIR / "recurrent_ppo")
)

print("\\n🚀 Début entraînement Recurrent PPO...")
start_time = time.time()
model_recurrent_ppo.learn(total_timesteps=TOTAL_TIMESTEPS, callback=logging_callback_recurrent_ppo, progress_bar=True)
recurrent_ppo_train_time = time.time() - start_time

history_recurrent_ppo = logging_callback_recurrent_ppo.get_history()
with open(LOGS_DIR / "recurrent_ppo_history.json", 'w') as f:
    json.dump(history_recurrent_ppo, f, indent=2)

model_recurrent_ppo.save(MODELS_DIR / "recurrent_ppo_final")

print(f"\\n✅ Recurrent PPO entraîné en {recurrent_ppo_train_time:.1f}s ({recurrent_ppo_train_time/60:.1f} min)")

results_recurrent_ppo_val = evaluate_policy_common(
    env=env_val,
    policy=lambda obs: model_recurrent_ppo.predict(obs, deterministic=True)[0],
    n_eval_episodes=100,
    model_name="Recurrent PPO",
    model_family="Policy-Based Advanced",
    split_name="validation"
)

results_recurrent_ppo_test = evaluate_policy_common(
    env=env_test,
    policy=lambda obs: model_recurrent_ppo.predict(obs, deterministic=True)[0],
    n_eval_episodes=100,
    model_name="Recurrent PPO",
    model_family="Policy-Based Advanced",
    split_name="test"
)

# %% [markdown]
# ## 🔄 Recurrent PPO - Mémoire Temporelle
#
# **Principe :** Utilise un réseau LSTM pour capturer les dépendances temporelles entre états
#
# **Architecture :**
# - État → LSTM(128) → Fully Connected → Actor/Critic
# - Le LSTM maintient un état caché qui capture l'historique
#
# **Avantages :**
# - Peut apprendre des patterns temporels
# - Utile si décisions passées influencent futures
# - Meilleure pour séquences longues
#
# **Attentes :** Performance potentiellement meilleure si patterns temporels existent

# %%
# =============================================================================
# SAC (Soft Actor-Critic) - Actions Continues Adaptées
# =============================================================================
print("\\n" + "="*80)
print("ENTRAINEMENT: SAC (Soft Actor-Critic) - Version Continue")
print(f"Device: {SB3_DEVICE}")
print("="*80)

# Création d'un environnement avec actions continues
class ContinuousPricingEnv(DynamicPricingEnv):
    """
    Version continue de l'environnement : l'agent choisit directement
    un multiplicateur continu dans [0.8, 1.5]
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Redéfinir l'espace d'action en continu
        self.action_space = spaces.Box(
            low=np.array([0.8], dtype=np.float32),
            high=np.array([1.5], dtype=np.float32),
            dtype=np.float32
        )
    
    def step(self, action):
        """Step avec action continue"""
        # L'action est maintenant un float dans [0.8, 1.5]
        multiplier = float(np.clip(action[0], 0.8, 1.5))
        
        # Récupération de la ligne actuelle
        row = self.data.iloc[self.current_index]
        base_price = row['Historical_Cost_of_Ride']
        proposed_price = base_price * multiplier
        
        # Simulation de la demande (identique)
        accepted = self._simulate_demand(
            base_price=base_price,
            proposed_price=proposed_price,
            ratio_riders_drivers=row['Ratio_Riders_Drivers'],
            loyalty_status=row['Customer_Loyalty_Status_encoded']
        )
        
        reward = proposed_price if accepted else 0.0
        
        self.episode_revenue += reward
        self.episode_accepted += int(accepted)
        self.current_step += 1
        
        terminated = self.current_step >= self.episode_length
        truncated = False
        
        if not terminated:
            self.current_index = (self.current_index + 1) % len(self.data)
            self.current_state = self._get_observation(self.current_index)
        
        info = {
            'episode_step': self.current_step,
            'episode_revenue': self.episode_revenue,
            'episode_accepted': self.episode_accepted,
            'acceptance_rate': self.episode_accepted / self.current_step if self.current_step > 0 else 0.0,
            'avg_revenue_per_step': self.episode_revenue / self.current_step if self.current_step > 0 else 0.0,
            'action': multiplier,
            'multiplier': multiplier,
            'accepted': accepted,
            'proposed_price': proposed_price
        }
        
        return self.current_state, reward, terminated, truncated, info

# Création des environnements continus
env_train_continuous = ContinuousPricingEnv(
    data=df_train,
    price_multipliers=PRICE_MULTIPLIERS,
    episode_length=EPISODE_LENGTH,
    demand_elasticity=3.0,
    random_state=GLOBAL_SEED
)

env_val_continuous = ContinuousPricingEnv(
    data=df_val,
    price_multipliers=PRICE_MULTIPLIERS,
    episode_length=EPISODE_LENGTH,
    demand_elasticity=3.0,
    random_state=GLOBAL_SEED + 1
)

env_test_continuous = ContinuousPricingEnv(
    data=df_test,
    price_multipliers=PRICE_MULTIPLIERS,
    episode_length=EPISODE_LENGTH,
    demand_elasticity=3.0,
    random_state=GLOBAL_SEED + 2
)

print(f"Environnements continus créés")
print(f"Action space: {env_train_continuous.action_space}")

# Callback pour SAC
logging_callback_sac = DetailedLoggingCallback(
    eval_env=env_val_continuous,
    eval_freq=EVAL_FREQ,
    n_eval_episodes=N_EVAL_EPISODES,
    verbose=1
)

# SAC Model
model_sac = SAC(
    "MlpPolicy",
    env_train_continuous,
    learning_rate=3e-4,
    buffer_size=100_000,
    learning_starts=1000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    ent_coef='auto',  # Coefficient d'entropie automatique
    target_entropy='auto',
    policy_kwargs=dict(
        net_arch=[256, 256],
        log_std_init=-3
    ),
    verbose=1,
    device=SB3_DEVICE,
    tensorboard_log=str(LOGS_DIR / "sac")
)

print("\\n🚀 Début entraînement SAC...")
start_time = time.time()
model_sac.learn(total_timesteps=TOTAL_TIMESTEPS, callback=logging_callback_sac, progress_bar=True)
sac_train_time = time.time() - start_time

history_sac = logging_callback_sac.get_history()
with open(LOGS_DIR / "sac_history.json", 'w') as f:
    json.dump(history_sac, f, indent=2)

model_sac.save(MODELS_DIR / "sac_final")

print(f"\\n✅ SAC entraîné en {sac_train_time:.1f}s ({sac_train_time/60:.1f} min)")

results_sac_val = evaluate_policy_common(
    env=env_val_continuous,
    policy=lambda obs: model_sac.predict(obs, deterministic=True)[0],
    n_eval_episodes=100,
    model_name="SAC",
    model_family="Deep RL Advanced (Continuous)",
    split_name="validation"
)

results_sac_test = evaluate_policy_common(
    env=env_test_continuous,
    policy=lambda obs: model_sac.predict(obs, deterministic=True)[0],
    n_eval_episodes=100,
    model_name="SAC",
    model_family="Deep RL Advanced (Continuous)",
    split_name="test"
)

# %% [markdown]
# ## 🎯 SAC (Soft Actor-Critic) - State-of-the-Art Continu
#
# **Principe :** Algorithme off-policy avec maximum entropy reinforcement learning
#
# **Caractéristiques clés :**
# - **Actions continues** : L'agent choisit un multiplicateur réel dans [0.8, 1.5]
# - **Maximum entropy** : Encourage l'exploration via terme d'entropie
# - **Twin Q-networks** : Réduit le biais d'overestimation
# - **Automatic entropy tuning** : Ajuste automatiquement le trade-off exploration/exploitation
#
# **Avantages :**
# - State-of-the-art pour actions continues
# - Très sample efficient
# - Excellente stabilité
# - Exploration naturelle
#
# **Comparaison avec discret :**
# - Version discrète (DQN/PPO) : Choisit parmi 8 actions fixes
# - Version continue (SAC) : Peut choisir n'importe quel multiplicateur (ex: 1.17x, 0.93x)
# - SAC peut trouver des prix plus fins et optimaux
#
# **Attentes :** Performance comparable ou supérieure aux modèles discrets avec plus de flexibilité

# %%
print("\\n" + "="*80)
print("✅ TOUS LES MODÈLES ENTRAÎNÉS (6 algorithmes)")
print("="*80)
print(f"DQN: {dqn_train_time/60:.1f} min")
print(f"A2C: {a2c_train_time/60:.1f} min")
print(f"PPO: {ppo_train_time/60:.1f} min")
print(f"Dueling DQN: {dueling_dqn_train_time/60:.1f} min")
print(f"Recurrent PPO: {recurrent_ppo_train_time/60:.1f} min")
print(f"SAC: {sac_train_time/60:.1f} min")
print(f"Total: {(dqn_train_time + a2c_train_time + ppo_train_time + dueling_dqn_train_time + recurrent_ppo_train_time + sac_train_time)/60:.1f} min")

# %% [markdown]
# ## 📊 Mise à Jour des Analyses avec les Nouveaux Modèles
#
# Maintenant que nous avons 6 algorithmes (au lieu de 3), nous allons :
# 1. Mettre à jour les courbes d'apprentissage comparatives
# 2. Analyser la convergence des 3 nouveaux modèles
# 3. Comparer l'efficacité des 6 algorithmes
# 4. Analyser le comportement des politiques apprises

# %%
# Consolidation des historiques (6 modèles)
histories_all = {
    'DQN': history_dqn,
    'A2C': history_a2c,
    'PPO': history_ppo,
    'Dueling DQN': history_dueling_dqn,
    'Recurrent PPO': history_recurrent_ppo,
    'SAC': history_sac
}

# Courbes d'apprentissage pour les 6 modèles
plot_learning_curves(
    histories=histories_all,
    save_path=FIGURES_DIR / 'learning_curves_comparison_all.png'
)

print("\\n✅ Courbes d'apprentissage (6 modèles) générées")

# %%
# Analyse de convergence des nouveaux modèles

# Dueling DQN
print("\\n" + "="*80)
print("ANALYSE DE CONVERGENCE: Dueling DQN")
print("="*80)
analyze_convergence(
    history=history_dueling_dqn,
    model_name="Dueling DQN",
    window=10,
    save_path=FIGURES_DIR / 'convergence_dueling_dqn.png'
)

# Recurrent PPO
print("\\n" + "="*80)
print("ANALYSE DE CONVERGENCE: Recurrent PPO")
print("="*80)
analyze_convergence(
    history=history_recurrent_ppo,
    model_name="Recurrent PPO",
    window=10,
    save_path=FIGURES_DIR / 'convergence_recurrent_ppo.png'
)

# SAC
print("\\n" + "="*80)
print("ANALYSE DE CONVERGENCE: SAC")
print("="*80)
analyze_convergence(
    history=history_sac,
    model_name="SAC",
    window=10,
    save_path=FIGURES_DIR / 'convergence_sac.png'
)

# %%
# Efficacité d'apprentissage (6 modèles)
compare_training_efficiency(
    histories=histories_all,
    save_path=FIGURES_DIR / 'training_efficiency_all.png'
)

print("\\n✅ Analyse d'efficacité (6 modèles) terminée")

# %%
# Analyse comportementale des nouveaux modèles

# Dueling DQN
print("\\n" + "="*80)
print("ANALYSE COMPORTEMENTALE: Dueling DQN")
print("="*80)
analyze_policy_behavior(
    env=env_test,
    model=model_dueling_dqn,
    model_name="Dueling DQN",
    n_episodes=10,
    save_path=FIGURES_DIR / 'policy_behavior_dueling_dqn.png'
)

# Recurrent PPO
print("\\n" + "="*80)
print("ANALYSE COMPORTEMENTALE: Recurrent PPO")
print("="*80)
analyze_policy_behavior(
    env=env_test,
    model=model_recurrent_ppo,
    model_name="Recurrent PPO",
    n_episodes=10,
    save_path=FIGURES_DIR / 'policy_behavior_recurrent_ppo.png'
)

# SAC (sur environnement continu)
print("\\n" + "="*80)
print("ANALYSE COMPORTEMENTALE: SAC")
print("="*80)
analyze_policy_behavior(
    env=env_test_continuous,
    model=model_sac,
    model_name="SAC",
    n_episodes=10,
    save_path=FIGURES_DIR / 'policy_behavior_sac.png'
)

# %% [markdown]
# ## 🔬 Comparaison Discret vs Continu
#
# **Analyse clé :** SAC (continu) vs DQN/PPO (discret)
#
# **Espace d'action :**
# - **Discret** : 8 choix fixes [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
# - **Continu** : N'importe quel multiplicateur dans [0.8, 1.5]
#
# **Avantages du continu (SAC) :**
# - Prix plus fins (ex: 1.17x au lieu de 1.1x ou 1.2x)
# - Potentiellement plus optimal
# - Exploration plus naturelle
#
# **Avantages du discret (DQN/PPO) :**
# - Plus simple à interpréter
# - Plus facile à implémenter en production
# - Actions explicites et compréhensibles
#
# **Question de recherche :** Le gain de finesse du continu compense-t-il la complexité ?
