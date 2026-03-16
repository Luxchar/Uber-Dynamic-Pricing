# 🚀 Projet Deep RL Dynamic Pricing pour Uber

## 📋 Vue d'Ensemble

Ce projet implémente et compare **6 algorithmes de Deep Reinforcement Learning** pour optimiser le dynamic pricing chez Uber. L'objectif est de maximiser le revenu tout en maintenant un bon taux d'acceptation client.

**Dataset :** 1 million de trajets synthétiques  
**Algorithmes :** DQN, A2C, PPO, Dueling DQN, Recurrent PPO, SAC  
**Analyses :** ~60 graphiques professionnels + Benchmark automatique

---

## 📚 Documentation

Le projet comprend 3 fichiers de documentation essentiels :

### 1. 🎯 [GUIDE_OPERATIONNEL.md](GUIDE_OPERATIONNEL.md)
**Pour qui :** Toute personne qui veut exécuter le projet

**Contenu :**
- Installation en 10 minutes
- Structure du projet
- Les 2 notebooks disponibles (3 ou 6 modèles)
- Exécution pas à pas
- Démo Streamlit
- Dépannage
- Checklist complète

**👉 Commencez par ce fichier pour prendre en main le projet**

---

### 2. 🔬 [EXPLICATIONS_TECHNIQUES.md](EXPLICATIONS_TECHNIQUES.md)
**Pour qui :** Personnes qui veulent comprendre l'implémentation technique

**Contenu :**
- Formalisation complète du MDP (État/Action/Récompense)
- Explication détaillée des 6 algorithmes
- Architectures des réseaux de neurones
- Protocole expérimental rigoureux
- Analyses avancées (convergence, sample efficiency, etc.)
- Justifications techniques de tous les choix

**👉 Lisez ce fichier pour comprendre comment tout fonctionne**

---

### 3. 🎤 [BRIEF_COACHING_ORAL.md](BRIEF_COACHING_ORAL.md)
**Pour qui :** Préparation à la soutenance orale

**Contenu :**
- Brief complet du projet pour ChatGPT
- 15 questions fréquentes avec réponses détaillées
- Messages clés pour la soutenance
- Phrases d'accroche
- Checklist avant soutenance
- Suggestions d'entraînement avec ChatGPT

**👉 Envoyez ce fichier à ChatGPT pour vous entraîner à l'oral**

---

## ⚡ Démarrage Ultra-Rapide

### Installation (5 min)
```bash
# Cloner le projet
cd Uber-Dynamic-Pricing

# Installer dépendances
pip install -r requirements.txt

# (Optionnel) GPU pour RTX 4080
pip install torch==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121

# Vérifier GPU
python check_gpu.py
```

### Exécution (30 min)
```bash
# Option 1 : Notebook 3 modèles (rapide - 15 min)
jupyter notebook research/notebooks/rl_dynamic_pricing_enriched.ipynb

# Option 2 : Notebook 6 modèles (complet - 30 min) ⭐
jupyter notebook research/notebooks/rl_dynamic_pricing_6_models.ipynb
```

Dans Jupyter : `Kernel > Restart & Run All`

### Démo Interactive
```bash
streamlit run app.py
```

---

## 🧠 Les 6 Algorithmes RL

| Algorithme | Type | Caractéristique Clé |
|------------|------|---------------------|
| **DQN** | Value-Based | Experience Replay + Target Network |
| **A2C** | Actor-Critic | Synchrone, rapide |
| **PPO** | Policy-Based | Clipping pour stabilité |
| **Dueling DQN** | Value-Based Advanced | Sépare V(s) et A(s,a) |
| **Recurrent PPO** | Policy + Memory | LSTM pour patterns temporels |
| **SAC** | Continuous Deep RL | Actions continues, Maximum Entropy |

---

## 📊 Structure du Projet

```
Uber-Dynamic-Pricing/
│
├── data/
│   ├── raw/dynamic_pricing.csv              # Dataset original
│   ├── processed/dynamic_pricing_1M.csv     # 1M lignes générées
│   └── generate_synthetic_data.py
│
├── src/utils/
│   ├── pricing_env.py                       # Environnement RL custom
│   ├── evaluation.py                        # Évaluation commune
│   └── advanced_analysis.py                 # Analyses approfondies
│
├── research/notebooks/
│   ├── rl_dynamic_pricing_enriched.ipynb    # 3 modèles (15 min)
│   └── rl_dynamic_pricing_6_models.ipynb    # 6 modèles (30 min) ⭐
│
├── research/results/                         # Généré après exécution
│   ├── models/                              # 6 modèles + best_model.zip
│   ├── figures/                             # ~60 graphiques PNG
│   └── logs/                                # Historiques JSON
│
├── app.py                                   # Démo Streamlit
├── requirements.txt                         # Dépendances Python
├── check_gpu.py                             # Vérification GPU
│
└── Documentation/
    ├── README.md                            # Ce fichier
    ├── GUIDE_OPERATIONNEL.md               # 👉 Guide d'exécution
    ├── EXPLICATIONS_TECHNIQUES.md          # 👉 Détails techniques
    └── BRIEF_COACHING_ORAL.md              # 👉 Préparation soutenance
```

---

## 🎯 Résultats Générés

### Après Exécution du Notebook

**Modèles entraînés :**
```
research/results/models/
├── dqn_final.zip
├── a2c_final.zip
├── ppo_final.zip
├── dueling_dqn_final.zip
├── recurrent_ppo_final.zip
├── sac_final.zip
└── best_model.zip (meilleur automatiquement)
```

**Graphiques (~60) :**
- Courbes d'apprentissage comparatives
- Analyses de convergence (6 modèles)
- Sample efficiency
- Comportement des politiques
- Et bien plus...

**Benchmark :**
- `benchmark_global.csv` : Comparaison complète des 9 modèles (6 RL + 3 baselines)
- Classement automatique selon Mean Reward Test

---

## ⏱️ Temps d'Exécution

| Configuration | Notebook 3 Modèles | Notebook 6 Modèles |
|---------------|--------------------|--------------------|
| **GPU (RTX 4080)** | 15 min | 30 min |
| **CPU** | 45 min | 90 min |

---

## 🛠️ Technologies Utilisées

- **Python 3.9+**
- **Gymnasium** : Environnements RL
- **Stable-Baselines3** : Implémentations RL state-of-the-art
- **sb3-contrib** : Algorithmes additionnels (Recurrent PPO)
- **PyTorch** : Deep Learning framework
- **Pandas / NumPy** : Manipulation de données
- **Matplotlib / Seaborn** : Visualisations
- **Streamlit** : Application démo interactive
- **Scikit-learn** : Preprocessing + Baselines ML

---

## 📖 Comment Utiliser Ce Projet

### 1️⃣ Pour Exécuter le Projet
👉 Lisez **[GUIDE_OPERATIONNEL.md](GUIDE_OPERATIONNEL.md)**

### 2️⃣ Pour Comprendre la Technique
👉 Lisez **[EXPLICATIONS_TECHNIQUES.md](EXPLICATIONS_TECHNIQUES.md)**

### 3️⃣ Pour Préparer la Soutenance
👉 Envoyez **[BRIEF_COACHING_ORAL.md](BRIEF_COACHING_ORAL.md)** à ChatGPT

---

## 🎓 Contributions Scientifiques

1. **Comparaison systématique** de 6 algorithmes RL sur le dynamic pricing
2. **Protocole expérimental rigoureux** et reproductible
3. **Analyses approfondies** : Convergence, sample efficiency, comportement
4. **Comparaison discret vs continu** (SAC vs autres)
5. **Code open-source** complet et documenté

---

## ⚠️ Limitations

**Assumées et documentées :**
- Modèle de demande simplifié (simulation avec élasticité)
- Dataset synthétique (distributions préservées)
- Features limitées (11 dimensions)
- Offline learning (pas de déploiement en ligne)

**Perspectives :**
- Offline RL avec données réelles complètes
- Features additionnelles (météo temps réel, événements)
- Multi-agent RL (concurrence plateformes)
- A/B testing en production

---

## 📞 Aide et Support

- **Problème d'installation ?** → Voir section "Dépannage" dans [GUIDE_OPERATIONNEL.md](GUIDE_OPERATIONNEL.md)
- **Question technique ?** → Voir [EXPLICATIONS_TECHNIQUES.md](EXPLICATIONS_TECHNIQUES.md)
- **Préparation orale ?** → Voir [BRIEF_COACHING_ORAL.md](BRIEF_COACHING_ORAL.md)

---

## ✅ Checklist Rapide

### Avant de Commencer
- [ ] Python 3.9+ installé
- [ ] `pip install -r requirements.txt` exécuté
- [ ] (Optionnel) GPU vérifié avec `python check_gpu.py`
- [ ] Dataset généré (existe déjà : `data/processed/dynamic_pricing_1M.csv`)

### Exécution
- [ ] Notebook choisi (3 ou 6 modèles)
- [ ] Exécuté avec succès (`Kernel > Restart & Run All`)
- [ ] Résultats consultés dans `research/results/`

### Démonstration
- [ ] Streamlit lancé (`streamlit run app.py`)
- [ ] Démo testée sur http://localhost:8501

---

## 🎉 Résumé

**Ce que fait ce projet :**
- Compare 6 algorithmes Deep RL pour le dynamic pricing
- Génère ~60 graphiques d'analyse professionnels
- Sélectionne automatiquement le meilleur modèle
- Fournit une démo Streamlit interactive

**Temps nécessaire :**
- Installation : 10 min
- Exécution : 15-30 min (selon notebook)
- Analyse : À votre rythme

**Documentation :**
- 3 fichiers MD essentiels (opérationnel, technique, oral)
- Code source commenté
- Notebooks exécutables

---

## 🚀 Pour Commencer Maintenant

**Étape 1 :** Lisez [GUIDE_OPERATIONNEL.md](GUIDE_OPERATIONNEL.md)

**Étape 2 :** Exécutez un notebook

**Étape 3 :** Analysez les résultats

**C'est tout ! Le projet est prêt à l'emploi. 🎓**

---

**Auteur :** Yann  
**Date :** Mars 2026  
**Contexte :** Projet académique Deep Reinforcement Learning

---

**⭐ Le projet est complet, documenté, et prêt pour la soutenance ! ⭐**
