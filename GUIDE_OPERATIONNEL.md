# 🚀 GUIDE OPÉRATIONNEL - Projet Deep RL Dynamic Pricing

## 📋 Vue d'Ensemble

**Projet :** Optimisation du Dynamic Pricing pour Uber avec Deep Reinforcement Learning  
**Dataset :** 1 million de trajets synthétiques  
**Objectif :** Comparer 6 algorithmes RL pour maximiser le revenu

---

## ⚡ Installation Rapide (10 min)

### 1. Prérequis
- Python 3.9+
- (Optionnel mais recommandé) GPU NVIDIA avec CUDA

### 2. Installation des Dépendances

```bash
# Dépendances de base
pip install -r requirements.txt

# GPU (optionnel - pour RTX 4080)
pip install torch==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121
```

### 3. Vérification GPU (si applicable)

```bash
python check_gpu.py
```

**Sortie attendue :**
- CUDA disponible : Oui
- Device actuel : cuda
- Nom GPU : NVIDIA GeForce RTX 4080

---

## 📂 Structure du Projet

```
Uber-Dynamic-Pricing/
│
├── data/
│   ├── raw/dynamic_pricing.csv              # Dataset original
│   ├── processed/dynamic_pricing_1M.csv     # 1M lignes générées
│   └── generate_synthetic_data.py           # Générateur
│
├── src/
│   └── utils/
│       ├── pricing_env.py                   # Environnement RL custom
│       ├── evaluation.py                    # Évaluation commune
│       └── advanced_analysis.py             # Analyses approfondies
│
├── research/
│   ├── notebooks/
│   │   ├── rl_dynamic_pricing_enriched.ipynb      # 3 modèles (15 min)
│   │   └── rl_dynamic_pricing_6_models.ipynb      # 6 modèles (30 min)
│   └── results/
│       ├── models/                          # Modèles entraînés
│       ├── figures/                         # Graphiques
│       └── logs/                            # Historiques JSON
│
├── app.py                                   # Démo Streamlit
├── requirements.txt                         # Dépendances
└── check_gpu.py                             # Vérification GPU
```

---

## 🎯 Les 2 Notebooks Disponibles

### Notebook 1 : 3 Modèles (Standard)
**Fichier :** `research/notebooks/rl_dynamic_pricing_enriched.ipynb`

**Modèles :**
- DQN (Deep Q-Network)
- A2C (Advantage Actor-Critic)
- PPO (Proximal Policy Optimization)

**Durée :** 15 min (GPU) | 45 min (CPU)

**Pour qui :** Projet académique standard, soutenance courte

---

### Notebook 2 : 6 Modèles (Avancé) ⭐
**Fichier :** `research/notebooks/rl_dynamic_pricing_6_models.ipynb`

**Modèles :**
- DQN, A2C, PPO (de base)
- Dueling DQN (architecture améliorée)
- Recurrent PPO (avec LSTM)
- SAC (actions continues)

**Durée :** 30 min (GPU) | 90 min (CPU)

**Pour qui :** Projet de recherche, comparaison avancée

---

## 🚀 Exécution Pas à Pas

### Étape 1 : Générer les Données (si nécessaire)

```bash
cd data
python generate_synthetic_data.py
```

**Sortie :** `data/processed/dynamic_pricing_1M.csv` (1M lignes, ~120 MB)

---

### Étape 2 : Lancer un Notebook

#### Option A : Notebook 3 Modèles (Rapide)
```bash
jupyter notebook research/notebooks/rl_dynamic_pricing_enriched.ipynb
```

#### Option B : Notebook 6 Modèles (Complet) ⭐
```bash
jupyter notebook research/notebooks/rl_dynamic_pricing_6_models.ipynb
```

---

### Étape 3 : Exécuter le Notebook

Dans Jupyter :
1. Cliquez sur `Kernel` → `Restart & Run All`
2. Attendez la fin de l'exécution (15-30 min selon le notebook)

**Le notebook fait automatiquement :**
- Chargement et exploration des données
- Création des environnements RL
- Entraînement des baselines non-RL
- Entraînement des modèles RL
- Génération des analyses et graphiques
- Benchmark global et sélection du meilleur modèle

---

### Étape 4 : Consulter les Résultats

#### Modèles Entraînés
```
research/results/models/
├── dqn_final.zip
├── a2c_final.zip
├── ppo_final.zip
├── dueling_dqn_final.zip      (si notebook 6 modèles)
├── recurrent_ppo_final.zip    (si notebook 6 modèles)
├── sac_final.zip              (si notebook 6 modèles)
└── best_model.zip             (meilleur sélectionné)
```

#### Graphiques
```
research/results/figures/
├── learning_curves_comparison*.png
├── convergence_[model].png
├── training_efficiency*.png
├── policy_behavior_[model].png
└── [~35-60 graphiques selon notebook]
```

#### Benchmark
```
research/results/benchmark_global.csv
research/results/benchmark_global.json
```

---

## 🌐 Démo Streamlit

### Lancer la Démo Interactive

```bash
streamlit run app.py
```

**URL :** http://localhost:8501

**Fonctionnalités :**
- Simulation de décision de pricing en temps réel
- Ajustement des paramètres (peak hours, ratio, loyalty)
- Visualisation de l'action choisie par le modèle
- Prédiction du prix optimisé

---

## 📊 Métriques et Résultats

### Métriques Évaluées
- **Mean Reward** : Revenu moyen par épisode
- **Acceptance Rate** : Taux d'acceptation des prix proposés
- **Avg Revenue per Episode** : Revenu moyen
- **Training Time** : Temps d'entraînement
- **Stability (Std Reward)** : Stabilité de l'apprentissage

### Classement Automatique
Le meilleur modèle est sélectionné automatiquement selon le **Mean Reward Test**.

---

## 🔧 Dépannage

### Problème : ImportError sur stable-baselines3
**Solution :**
```bash
pip install stable-baselines3[extra] sb3-contrib
```

### Problème : Pas de GPU détecté
**Solution :**
1. Vérifier installation CUDA : `nvidia-smi`
2. Réinstaller PyTorch avec CUDA :
```bash
pip uninstall torch
pip install torch==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121
```

### Problème : Notebook trop lent
**Solutions :**
- Utiliser le notebook 3 modèles au lieu de 6
- Réduire `TOTAL_TIMESTEPS` dans le notebook (section Configuration)
- Utiliser un GPU

### Problème : Out of Memory (GPU)
**Solution :**
- Réduire `batch_size` des modèles
- Réduire `buffer_size` de DQN/SAC
- Utiliser CPU (plus lent mais fonctionne)

---

## 📦 Fichiers Importants

### Code Source
| Fichier | Description |
|---------|-------------|
| `src/utils/pricing_env.py` | Environnement Gymnasium custom |
| `src/utils/evaluation.py` | Fonction d'évaluation commune |
| `src/utils/advanced_analysis.py` | Analyses approfondies |
| `app.py` | Application Streamlit |

### Configuration
| Fichier | Description |
|---------|-------------|
| `requirements.txt` | Toutes les dépendances Python |
| `check_gpu.py` | Script vérification GPU |

### Données
| Fichier | Description |
|---------|-------------|
| `data/raw/dynamic_pricing.csv` | Dataset original (~60 KB) |
| `data/processed/dynamic_pricing_1M.csv` | Dataset 1M lignes (~120 MB) |

---

## ⏱️ Temps d'Exécution

### Avec GPU (RTX 4080)
| Tâche | Durée |
|-------|-------|
| Génération données | 2 min |
| Notebook 3 modèles | 15 min |
| Notebook 6 modèles | 30 min |
| Total (6 modèles) | ~35 min |

### Sans GPU (CPU)
| Tâche | Durée |
|-------|-------|
| Génération données | 5 min |
| Notebook 3 modèles | 45 min |
| Notebook 6 modèles | 90 min |
| Total (6 modèles) | ~95 min |

---

## ✅ Checklist Rapide

### Avant de Commencer
- [ ] Python 3.9+ installé
- [ ] `pip install -r requirements.txt` exécuté
- [ ] (Optionnel) GPU configuré et vérifié
- [ ] Dataset généré (`dynamic_pricing_1M.csv` existe)

### Exécution
- [ ] Notebook choisi (3 ou 6 modèles)
- [ ] Notebook exécuté avec succès
- [ ] Résultats générés dans `research/results/`
- [ ] Benchmark consulté

### Démo
- [ ] Streamlit lancé (`streamlit run app.py`)
- [ ] Démo fonctionnelle sur localhost:8501

---

## 🎯 Workflow Recommandé

### Pour Tester Rapidement
1. Installer dépendances
2. Exécuter notebook 3 modèles (15 min)
3. Consulter résultats
4. Tester démo Streamlit

### Pour Projet Complet
1. Installer dépendances + GPU
2. Générer données (si nécessaire)
3. Exécuter notebook 6 modèles (30 min)
4. Analyser graphiques et benchmark
5. Préparer démo Streamlit
6. Préparer présentation avec résultats

---

## 📞 Commandes Essentielles

```bash
# Installation
pip install -r requirements.txt

# Vérifier GPU
python check_gpu.py

# Générer données
cd data && python generate_synthetic_data.py

# Lancer notebook (choisir un)
jupyter notebook research/notebooks/rl_dynamic_pricing_enriched.ipynb
jupyter notebook research/notebooks/rl_dynamic_pricing_6_models.ipynb

# Lancer démo
streamlit run app.py
```

---

## 🎉 Résumé

**Installation :** 10 min  
**Exécution notebook :** 15-30 min selon choix  
**Résultats :** Modèles entraînés + ~60 graphiques + Benchmark  
**Démo :** Application Streamlit interactive  

**Le projet est prêt à être exécuté et démontré ! 🚀**
