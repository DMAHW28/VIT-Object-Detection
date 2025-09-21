# Détection d'objet avec ViT

## Description
Ce projet implémente un Vision Transformer (ViT) from scratch pour la détection d’objets (avions, hélicoptères et motobikers) ✈️🚁🏍️.
L'implémentation se base sur le papier [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929).

<img src="assets/ViT.png" alt="transformer" width="700"/> 

## Structure du projet
- `src/` : code source (modèle, entraînement, outils, data preprocessing)
- `models/` : modèles enregistrés (.pth)
- `database/` : données ou bases utilisées (exclues du dépôt)
- `evaluate.ipynb` : notebook d’évaluation des performances
- `train.ipynb` : notebook d’entraînement

## Installation
Cloner le dépôt et installer les dépendances :

```bash
pip install -r requirement.txt
```

##  Dataset

Télécharger le dataset :

```bash
python src/download_dataset.py
```

## Modèle et performances
Le modèle Vit-Tiny a été entraîné, les performances sur le jeu de test sont :

- Accuracy : 94 %
- IoU: 0.8726
- DIoU: 0.8717
- GIoU: 1.6045
- F1-score: 0.87

## Détection
![Description de l'image](assets/vit_detection.png)

## Technologies utilisées
 - Python 3.10
 - PyTorch
 - NumPy
 - PIL
 - albumentations (Data Augmentation)