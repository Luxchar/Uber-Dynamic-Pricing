# 🔬 EXPLICATIONS TECHNIQUES - Deep RL Dynamic Pricing

## 📊 Vue d'Ensemble Technique

**Problème :** Optimisation du pricing dynamique pour maximiser le revenu tout en maintenant un bon taux d'acceptation

**Approche :** Deep Reinforcement Learning avec comparaison de 6 algorithmes

**Dataset :** 1M trajets avec 11 features (pics, ratio offre/demande, météo, etc.)

---

## 🧠 Formalisation du Problème RL

### Markov Decision Process (MDP)

#### **État (State) - 11 dimensions**

```python
State = [
    Number_of_Riders,                    # Nombre de demandeurs
    Number_of_Drivers,                   # Nombre de conducteurs
    Location_Category_encoded,           # Zone géographique (0-2)
    Customer_Loyalty_Status_encoded,     # Fidélité client (0-2)
    Number_of_Past_Rides,               # Historique client
    Average_Ratings,                    # Note moyenne
    Time_of_Booking_encoded,            # Créneau horaire (0-3)
    Vehicle_Type_encoded,               # Type véhicule (0-1)
    Expected_Ride_Duration,             # Durée estimée
    Historical_Cost_of_Ride,            # Prix de base
    Ratio_Riders_Drivers               # Ratio offre/demande
]
```

**Normalisation :** StandardScaler sur features numériques, LabelEncoder sur catégorielles

---

#### **Action (Action Space)**

##### Discret (DQN, A2C, PPO, Dueling DQN, Recurrent PPO)
```python
action_space = spaces.Discrete(8)

# Mapping action → multiplicateur
PRICE_MULTIPLIERS = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

# Action 0 → 0.8x (discount 20%)
# Action 4 → 1.2x (surge 20%)
# Action 7 → 1.5x (surge 50%)
```

**Prix final :** `proposed_price = Historical_Cost_of_Ride × PRICE_MULTIPLIERS[action]`

##### Continu (SAC)
```python
action_space = spaces.Box(low=0.8, high=1.5, shape=(1,), dtype=float32)

# L'agent choisit directement le multiplicateur
# Exemple: 1.173x (prix très fin)
```

---

#### **Récompense (Reward)**

```python
def reward_function(proposed_price, accepted):
    if accepted:
        return proposed_price  # Revenue
    else:
        return 0.0  # Pas de revenu
```

**Mécanisme d'acceptation :** Simulation via fonction de demande

```python
def _simulate_demand(base_price, proposed_price, ratio_riders_drivers, loyalty_status):
    # Élasticité de la demande
    price_ratio = proposed_price / base_price
    elasticity_effect = np.exp(-DEMAND_ELASTICITY * (price_ratio - 1.0))
    
    # Influence de l'offre/demande
    ratio_effect = 1.0 + 0.5 * (ratio_riders_drivers - 1.0)
    
    # Bonus fidélité
    loyalty_effect = 1.0 + 0.1 * loyalty_status
    
    # Probabilité d'acceptation
    acceptance_prob = elasticity_effect * ratio_effect * loyalty_effect
    acceptance_prob = np.clip(acceptance_prob, 0.0, 1.0)
    
    # Décision aléatoire
    return np.random.rand() < acceptance_prob
```

**Paramètres :**
- `DEMAND_ELASTICITY = 3.0` : Sensibilité au prix (plus élevé = plus sensible)

---

#### **Transition (Dynamics)**

```python
def step(action):
    # 1. Calculer le prix proposé
    multiplier = PRICE_MULTIPLIERS[action]
    proposed_price = base_price * multiplier
    
    # 2. Simuler acceptation
    accepted = _simulate_demand(base_price, proposed_price, ...)
    
    # 3. Calculer récompense
    reward = proposed_price if accepted else 0.0
    
    # 4. Passer à l'état suivant
    next_state = get_next_observation()
    
    # 5. Épisode terminé ?
    terminated = (current_step >= episode_length)
    
    return next_state, reward, terminated, truncated, info
```

**Épisode :** 100 décisions de pricing consécutives

---

## 🤖 Les 6 Algorithmes RL Implémentés

### 1. DQN (Deep Q-Network)

**Type :** Value-Based, Off-Policy

**Architecture :**
```python
Input (11) → Dense(128, ReLU) → Dense(128, ReLU) → Output(8)
                                                      ↓
                                                Q-values pour 8 actions
```

**Principe :**
- Apprend Q(s, a) : valeur de chaque action dans chaque état
- Choisit action avec Q-value maximal
- Experience Replay : stocke transitions (s, a, r, s') dans buffer
- Target Network : réseau cible pour stabilité

**Hyperparamètres clés :**
```python
learning_rate = 1e-4
buffer_size = 100_000
batch_size = 32
gamma = 0.99  # Facteur de discount
tau = 0.005   # Soft update target network
exploration_fraction = 0.1
exploration_final_eps = 0.05
```

**Update Rule :**
```
Target: y = r + γ × max_a' Q_target(s', a')
Loss: MSE(Q(s, a), y)
```

---

### 2. A2C (Advantage Actor-Critic)

**Type :** Actor-Critic, On-Policy

**Architecture :**
```
Input (11) → Shared Encoder (128, 128)
              ↓                    ↓
         Actor (π)             Critic (V)
           (8)                    (1)
```

**Principe :**
- **Actor** : Politique π(a|s) (probabilités des actions)
- **Critic** : Value function V(s) (valeur de l'état)
- **Advantage** : A(s,a) = Q(s,a) - V(s)
- Synchrone : update après chaque rollout

**Hyperparamètres clés :**
```python
learning_rate = 7e-4
n_steps = 5      # Rollout length
gamma = 0.99
gae_lambda = 1.0
ent_coef = 0.01  # Coefficient entropie (exploration)
vf_coef = 0.5    # Coefficient value function
```

**Update Rules :**
```
Actor Loss: -log π(a|s) × A(s,a) - β × H[π]
Critic Loss: MSE(V(s), R_t)
```

---

### 3. PPO (Proximal Policy Optimization)

**Type :** Policy-Based, On-Policy

**Architecture :**
```
Input (11) → Shared (128, 128) → Actor (8) + Critic (1)
```

**Principe :**
- Optimisation de politique avec clipping pour stabilité
- Empêche des changements trop brutaux de politique
- Multiple epochs sur chaque batch de données

**Hyperparamètres clés :**
```python
learning_rate = 3e-4
n_steps = 2048
batch_size = 64
n_epochs = 10
gamma = 0.99
gae_lambda = 0.95
clip_range = 0.2  # Clipping PPO
ent_coef = 0.0
```

**Objective Function (Clipped) :**
```
L_CLIP = min(
    ratio × A,
    clip(ratio, 1-ε, 1+ε) × A
)

où ratio = π_new(a|s) / π_old(a|s)
```

---

### 4. Dueling DQN

**Type :** Value-Based Advanced, Off-Policy

**Architecture Dueling :**
```
Input (11) → Shared (256, 256)
              ↓              ↓
         Value Stream   Advantage Stream
            V(s)           A(s, a)
              ↓              ↓
         Q(s,a) = V(s) + (A(s,a) - mean(A(s,·)))
```

**Principe :**
- Sépare l'estimation de la valeur intrinsèque de l'état (V) et l'avantage de chaque action (A)
- Meilleure généralisation : V(s) appris même si certaines actions rarement prises
- Converge souvent plus vite que DQN standard

**Hyperparamètres :**
```python
policy_kwargs = dict(
    net_arch=[256, 256],
    dueling=True  # Active architecture Dueling
)
# Autres paramètres identiques à DQN
```

**Formule Q-value :**
```
Q(s,a) = V(s) + (A(s,a) - 1/|A| × Σ A(s,a'))
```

---

### 5. Recurrent PPO

**Type :** Policy-Based + Memory, On-Policy

**Architecture LSTM :**
```
Input (11) → LSTM(128) → Fully Connected
              ↓              ↓
         Hidden State    Actor (8) + Critic (1)
```

**Principe :**
- Réseau récurrent (LSTM) pour capturer dépendances temporelles
- Maintient un état caché qui "mémorise" l'historique
- Utile si patterns temporels (ex: pics répétés, tendances)

**Hyperparamètres :**
```python
policy = "MlpLstmPolicy"
policy_kwargs = dict(
    lstm_hidden_size=128,
    n_lstm_layers=1,
    enable_critic_lstm=True  # LSTM aussi pour critic
)
# Autres paramètres identiques à PPO
```

**Avantage :**
- Peut apprendre "si dernier prix élevé ET accepté → continuer surge"
- Capture seasonality et patterns récurrents

---

### 6. SAC (Soft Actor-Critic)

**Type :** Deep RL Continuous, Off-Policy

**Architecture :**
```
Input (11) → Actor → μ(s), σ(s) → action ~ N(μ, σ)
                                      ↓
Input (11) + action → Twin Q-Networks → Q1(s,a), Q2(s,a)
```

**Principe :**
- **Maximum Entropy RL** : Maximise reward + entropie de la politique
- **Twin Q-Networks** : Réduit biais d'overestimation (prend min des 2 Q)
- **Actions continues** : Peut choisir n'importe quel multiplicateur dans [0.8, 1.5]
- **Automatic Entropy Tuning** : Ajuste automatiquement exploration/exploitation

**Hyperparamètres :**
```python
learning_rate = 3e-4
buffer_size = 100_000
batch_size = 256
tau = 0.005
gamma = 0.99
ent_coef = 'auto'  # Ajustement automatique
target_entropy = 'auto'
policy_kwargs = dict(
    net_arch=[256, 256],
    log_std_init=-3
)
```

**Objective Function :**
```
J_π = E[r + γ × (Q(s',a') - α × log π(a'|s'))]

où α est le coefficient d'entropie (auto-tuned)
```

**Environnement Spécial : ContinuousPricingEnv**
```python
action_space = spaces.Box(low=0.8, high=1.5, shape=(1,))
# L'agent choisit directement le multiplicateur
```

---

## 🔬 Protocole Expérimental

### Split des Données

```python
train_size = 0.70  # 700k samples
val_size = 0.15    # 150k samples
test_size = 0.15   # 150k samples

# Shuffle avec seed fixe pour reproductibilité
np.random.seed(42)
```

### Configuration d'Entraînement

```python
GLOBAL_SEED = 42
TOTAL_TIMESTEPS = 100_000
EPISODE_LENGTH = 100
EVAL_FREQ = 5000
N_EVAL_EPISODES = 50
```

### Fonction d'Évaluation Commune

```python
def evaluate_policy_common(env, policy, n_eval_episodes, model_name):
    """
    Évalue TOUS les modèles (RL et non-RL) avec le même protocole
    """
    episode_rewards = []
    episode_acceptances = []
    
    for episode in range(n_eval_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_accepted = 0
        done = False
        
        while not done:
            action = policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_accepted += info['accepted']
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        episode_acceptances.append(episode_accepted)
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_acceptance': np.mean(episode_acceptances) / EPISODE_LENGTH,
        'model_name': model_name,
        ...
    }
```

**Importance :** Tous les modèles sont évalués de la même façon (fair comparison)

---

## 📊 Baselines Non-RL

### 1. Fixed Price (Multiplicateur = 1.0x)

```python
def fixed_price_policy(obs):
    return 2  # Action 2 → multiplicateur 1.0x
```

**Rôle :** Baseline naïve (pas de changement de prix)

### 2. Simple Heuristic

```python
def simple_heuristic_policy(obs):
    ratio = obs[10]  # Ratio_Riders_Drivers
    peak_hour = obs[6] in [1, 2]  # Time_of_Booking
    
    if ratio > 1.5 and peak_hour:
        return 6  # 1.4x (surge fort)
    elif ratio > 1.2:
        return 4  # 1.2x (surge modéré)
    elif ratio < 0.8:
        return 0  # 0.8x (discount)
    else:
        return 2  # 1.0x (standard)
```

**Rôle :** Heuristique basée sur domaine expertise

### 3. Supervised ML → Greedy Policy

```python
# 1. Entraîner modèle supervisé (RandomForest)
X = features
y = Expected_Ride_Duration (proxy pour demande)

rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)

# 2. Pour chaque état, tester les 8 actions et choisir celle avec meilleur score prédit
def supervised_greedy_policy(obs):
    best_action = None
    best_score = -inf
    
    for action in range(8):
        score = rf_model.predict([obs])[0] * PRICE_MULTIPLIERS[action]
        if score > best_score:
            best_score = score
            best_action = action
    
    return best_action
```

**Rôle :** Approche ML classique (non-RL) pour comparaison

---

## 📈 Analyses Avancées

### 1. Courbes d'Apprentissage

```python
def plot_learning_curves(histories):
    """
    Compare évolution du reward moyen pendant l'entraînement
    """
    for model_name, history in histories.items():
        plt.plot(history['timesteps'], history['mean_rewards'], label=model_name)
```

**Métriques :**
- Mean Reward vs Timesteps
- Convergence visuelle
- Stabilité (variance)

### 2. Analyse de Convergence

```python
def analyze_convergence(history, window=10):
    """
    Compare performance début vs fin d'entraînement
    """
    initial_rewards = rewards[:window]
    final_rewards = rewards[-window:]
    
    improvement = (mean(final) - mean(initial)) / mean(initial) * 100
    stability = std(final_rewards)
```

**Métriques :**
- Amélioration absolue (%)
- Stabilité finale (std)
- Distribution début vs fin

### 3. Sample Efficiency

```python
def compare_training_efficiency(histories):
    """
    Quel modèle atteint le meilleur reward avec le moins de timesteps ?
    """
    threshold = 0.9 * max_reward_all_models
    
    for model, history in histories.items():
        timesteps_to_threshold = find_first_above(history, threshold)
```

**Métrique clé :** Timesteps nécessaires pour atteindre 90% du reward final

### 4. Comportement des Politiques

```python
def analyze_policy_behavior(env, model, n_episodes):
    """
    Analyse quelles actions sont choisies dans quels contextes
    """
    actions_chosen = []
    contexts = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            actions_chosen.append(action)
            contexts.append({
                'peak': obs[6],
                'ratio': obs[10],
                'loyalty': obs[3]
            })
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
```

**Analyses :**
- Distribution des actions
- Actions vs pic/normal
- Actions vs ratio offre/demande
- Acceptance rate par action
- Revenue moyen par action

---

## 🏆 Benchmark Global

### Métriques Comparées

```python
benchmark_metrics = {
    'Model Name': str,
    'Family': str,  # Value-Based / Actor-Critic / Policy-Based / etc.
    'Mean Reward (Val)': float,
    'Std Reward (Val)': float,
    'Mean Reward (Test)': float,
    'Std Reward (Test)': float,
    'Acceptance Rate (Test)': float,
    'Avg Revenue per Episode (Test)': float,
    'Training Time (s)': float
}
```

### Classement Automatique

```python
# Tri par Mean Reward Test (décroissant)
benchmark_df = benchmark_df.sort_values('Mean Reward (Test)', ascending=False)

# Meilleur modèle
best_model_name = benchmark_df.iloc[0]['Model Name']
```

### Sauvegarde

```python
# CSV pour analyse
benchmark_df.to_csv('benchmark_global.csv', index=False)

# JSON pour réutilisation
benchmark_dict.to_json('benchmark_global.json', indent=2)

# Modèle sauvegardé
best_model.save('best_model.zip')
```

---

## 🔍 Justifications Techniques

### Pourquoi RL plutôt que ML Supervisé ?

**Problème séquentiel :**
- Les décisions de pricing impactent les états futurs
- Un prix élevé aujourd'hui peut réduire la demande demain
- RL optimise la récompense cumulative (long terme)

**ML supervisé :**
- Prédit un label fixe (ex: prix optimal instantané)
- Ne considère pas l'impact futur
- Nécessite labels parfaits (qu'on n'a pas)

### Pourquoi 6 Algorithmes ?

**Diversité des approches :**
- **Value-Based (DQN, Dueling DQN)** : Apprend Q(s,a), choisit max
- **Actor-Critic (A2C)** : Balance actor et critic
- **Policy-Based (PPO, Recurrent PPO)** : Optimise directement π(a|s)
- **Continuous (SAC)** : Espace d'action continu

**Comparaison rigoureuse :**
- Pas de "one size fits all" en RL
- Chaque algo a ses forces/faiblesses
- Le meilleur dépend du problème

### Pourquoi Environnement Custom ?

**Spécificités du pricing :**
- Récompense = revenue si accepté, 0 sinon
- Simulation de la demande avec élasticité
- Features métier (loyalty, peak hours, ratio)

**Pas d'environnement Gymnasium standard pour ce cas d'usage**

### Pourquoi Simulation de la Demande ?

**Problème :** Dataset ne contient pas "si j'avais proposé prix X, aurait-il été accepté ?"

**Solution :** Modèle de demande semi-réaliste
- Élasticité prix (exp(-3 × (price_ratio - 1)))
- Influence ratio offre/demande
- Bonus fidélité

**Limitation assumée :** Modèle simplifié, mais permet l'apprentissage RL

---

## 🎯 Résumé Technique

**Problème :** MDP avec state (11D), action (8 discrètes ou continu [0.8,1.5]), reward (revenue)

**Algorithmes :** 6 (DQN, A2C, PPO, Dueling DQN, Recurrent PPO, SAC)

**Protocole :** Train/Val/Test (70/15/15), évaluation commune, 100k timesteps

**Baselines :** Fixed price, Heuristic, Supervised ML

**Analyses :** Courbes apprentissage, convergence, sample efficiency, comportement politiques

**Résultat :** Benchmark automatique, sélection best model, ~60 graphiques

**Technologies :** Python, Gymnasium, Stable-Baselines3, PyTorch, Pandas, Matplotlib

---

**Le projet combine rigueur académique et implémentation pratique du Deep RL ! 🚀**
