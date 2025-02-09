# README

## Description 🚀🔥✅

Ce projet implémente des pipelines d'entraînement et d'inférence utilisant YOLO pour la détection d'objets. Les modèles et datasets sont gérés via Picsellia. 🎯🔍📊

---

## Prérequis 🛠️⚡📌

Avant d'exécuter le projet, assurez-vous d'avoir installé les dépendances et configuré les fichiers de paramètres. ✅🔧📂

### 1. Installer les dépendances 📥⚙️🐍

Utilisez `pip` pour installer les bibliothèques requises :

```bash
pip install -r requirements.txt
```

### 2. Configurer les fichiers nécessaires 📝🔑🛠️

#### `settings.toml` 📄🛠️🗂️

Ce fichier doit contenir les informations de votre dataset et votre groupe :

```toml
dataset_uuid = ""
groupe_name = ""
```

#### `.secrets.toml` 🔒🔑📁

Ajoutez vos clés API et informations Picsellia :

```toml
api_key = "VOTRE_API_KEY"
workspace_name = "VOTRE_WORKSPACE"
```

---

## Exécution des pipelines 🚀🛠️📊

### 1. Pipeline d'entraînement 🏋️‍♂️📈🎯

Pour entraîner le modèle YOLO avec votre dataset, exécutez :

```bash
python main.py --train
```

Cette commande :

- Télécharge et prépare les données, 📥🗂️⚡
- Structure le dataset, 🔄📊🛠️
- Entraîne un modèle YOLO avec les hyperparamètres définis dans `hyperparameters.yaml`, 🤖📈📂
- Stocke et enregistre le modèle sur Picsellia. 📤🔗✅

### 2. Pipeline d'inférence 🔍🤖📷

L'inférence peut être réalisée sur une image, une vidéo ou via la webcam. 🎥📸🖥️

#### Inférence sur une image 🖼️📊✅

```bash
python main.py --inference --image --source <chemin_de_l_image>
```

#### Inférence sur une vidéo 🎥🎞️🎯

```bash
python main.py --inference --video --source <chemin_de_la_vidéo>
```

#### Inférence en temps réel via webcam 📹🔍🤖

```bash
python main.py --inference --webcam
```

---

## Hyperparamètres 🎯⚙️📊

Le fichier `hyperparameters.yaml` permet de modifier les paramètres d'entraînement, tels que :

- Nombre d'epochs (`epochs`) 📆🔄📈
- Taille de batch (`batch`) 📦⚡🔍
- Optimiseur (`optimizer`) 🤖🔢⚙️
- Taux d'apprentissage (`lr0`) 🔥📊📈
- Augmentations (`mosaic`, `mixup`) 🎨🔄🖼️

---

## Pré-commit Hooks 🛠️✅📏

Le projet utilise `pre-commit` pour assurer la qualité du code. 🎯📋⚡

### Installation des hooks 📥⚙️📌

```bash
pre-commit install
```

### Exécution manuelle des hooks 🔄📝✅

```bash
pre-commit run --all-files
```

---

## Structure du projet 📂📊🛠️

```
├── config.py
├── main.py
├── pre-commit-config.yaml
├── requirements.txt
├── settings.toml
├── .secrets.toml
├── src/
│   ├── model_tuner.py
│   ├── Picsellia/
│   │   ├── PicselliaCallback.py
│   │   ├── PicselliaConfig.py
│   ├── Pipelines/
│   │   ├── Train/
│   │   │   ├── train_pipeline.py
│   │   │   ├── data_preparation.py
│   │   ├── Inference/
│   │   │   ├── inference_pipeline.py
```

---

## Remarque 🔎📌⚡

- Assurez-vous que votre environnement Picsellia est bien configuré. 🛠️✅📂
- Vérifiez que `api_key` et `workspace_name` sont correctement renseignés dans `.secrets.toml`. 🔑📋📝
- Testez vos pipelines avant le déploiement final. 🏁🚀✅
