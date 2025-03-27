# AeroSafe - Prototype IA

## 📌 Introduction
AeroSafe est un projet de drone intelligent destiné à assister les équipes de secours en détectant les victimes dans des environnements complexes grâce à l'intelligence artificielle et la vision par ordinateur.

Ce dépôt contient la première version du **prototype IA** d'AeroSafe, incluant un pipeline de données, un modèle initial et une structure de projet versionnée sous Git.

---

## 📅 Semaine 5 - Développement des bases de l'IA

### 🎯 Objectifs
- Construire un **pipeline de données** pour préparer les entrées du modèle IA.
- Développer et tester un **modèle simple** avec Yolo et OpenCV.
- Mettre en place un **dépôt Git organisé** pour le suivi du projet.

---

## 🏗️ Architecture du Projet
```
📁 data
│── 📁 sample_videos
│── 📁 docs
│   │── 📄 business-model-canvas.pdf
│   │── 📄 Introduction du Projet: Drones ...
│   │── 📄 Sans nom 1.pdf
│
📁 my-env (Environnement virtuel Python)
│
📁 notebooks (Dossiers pour les notebooks Jupyter)
│
📁 output (Résultats des tests YOLO)
│
📁 runs (Résultats des inférences YOLO)
│
📁 src (Code source principal)
│── 📁 configs (Fichiers de configuration)
│── 📁 models (Modèles entraînés ou architecture)
│── 📄 app.py (Script principal de l'application)
│── 📄 data_pipeline.py (Pipeline de traitement des données)
│── 📄 frame.py (Gestion des frames vidéo)
│── 📄 mediapipe.py (Utilisation de MediaPipe pour la détection)
│
📄 .gitignore (Fichier pour exclure certains fichiers dans Git)
📄 README.md (Documentation du projet)
📄 requirements.txt (Liste des dépendances Python)

```

---

## 🔹 1. Installation & Configuration
### 🔧 Prérequis
- Python 3.8+
- Git
- Environnement virtuel (recommandé : `venv` ou `conda`)
- Docker(pour la mimse en production)
### 🚀 Installation
```bash
# Cloner le dépôt
git clone https://github.com/ton-repo/AeroSafe.git
cd AeroSafe

# Créer et activer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Pour Mac/Linux
venv\Scripts\activate     # Pour Windows

# Installer les dépendances
pip install -r requirements.txt
```

---

## 🔹 2. Pipeline de données
### 📌 Objectif
Prétraiter les images et vidéos en vue de l'entraînement du modèle IA.

### 📜 Étapes
1. **Collecte des données** (images, vidéos, métadonnées GPS)
2. **Nettoyage** (suppression du bruit, redimensionnement)
3. **Normalisation** (standardisation des valeurs)
4. **Augmentation** (création de variations des images)
5. **Stockage structuré** (système de fichiers ou base de données)

### 🚀 Exécution
```bash
python src/data_pipeline.py
```

---

## 🔹 3. Modèle IA
### 📌 Objectif
Tester une première version du modèle de détection de victimes avec un algorithme simple avant d’intégrer YOLO, Mediapipe ou ViTPose.

### 📜 Modèles testés
YoloV8x
Yolo12x
Mediapipe


### 🚀 Entraînement & Évaluation
```bash
python src/model_train.py
python src/model_eval.py
```

---

## 🔹 4. Versionnement avec Git
### 📜 Branches principales
- `main` → Code stable
- `develop` → Fonctionnalités en cours
- `feature/data-pipeline` → Développement du pipeline de données
- `feature/model-test` → Expérimentation des modèles IA

### 🚀 Commandes utiles
```bash
# Vérifier l’état du dépôt
git status

# Ajouter des modifications
git add .

git commit -m "Ajout du pipeline de données"

# Pousser sur GitHub
git push origin feature/data-pipeline
```

---

## 📦 Livrables attendus
✔️ **Prototype IA fonctionnel**
✔️ **Dépôt Git structuré**
✔️ **Documentation claire et détaillée**

---

## 🔜 Prochaines étapes
📌 Sélection du **meilleur modèle IA**
📌 Intégration de **données réelles** des drones
📌 Expérimentation avec **algorithmes avancés** (YOLO, Mediapipe, ViTPose)


