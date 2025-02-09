# Rapport d'Expérimentation

## Introduction 📌

Ce rapport détaille les expérimentations menées dans le cadre du projet d'entraînement de modèles YOLO pour la détection d'objets.

---

## Expérimentations Conduites 🛠️

Nous avons mené plusieurs expérimentations afin d'obtenir un modèle performant. Nos tests ont porté sur différentes configurations d'hyperparamètres et sur l'utilisation de techniques de data augmentation. Nous avons également exploré l'effet du tuning automatique des hyperparamètres afin d'améliorer la convergence du modèle et d'optimiser sa capacité à généraliser sur de nouvelles données.

### 1. Configuration de Base 🔧

Nous avons commencé par entraîner notre modèle YOLO11n avec des hyperparamètres standards. Cette configuration a permis d'atteindre le **meilleur fitness** de **0.30**, ce qui en fait notre référence principale pour les comparaisons ultérieures.

```yaml
model: "yolo11n.pt"
epochs: 150
patience: 20
batch: 16
imgsz: 640
device: "0"
workers: 8
exist_ok: True
optimizer: "AdamW"
lr0: 0.001
```

Le modèle a convergé correctement et a montré des résultats relativement stables sur les données de validation, bien qu'il soit encore loin d'une performance optimale pour une application pratique.

### 2. Test avec Data Augmentation et Hyperparamètres Avancés 📊

Afin d'améliorer les performances du modèle, nous avons testé des stratégies plus complexes, incluant la **data augmentation** (`mosaic`, `mixup`) ainsi que des modifications des hyperparamètres liés à l'optimisation et au prétraitement des données.

```yaml
epochs: 150
patience: 20
batch: 32
imgsz: 640
workers: 8
exist_ok: true
cache: true
optimizer: "AdamW"
lr0: 0.0005
weight_decay: 0.0005
warmup_epochs: 3
mosaic: true
mixup: true
```

Cependant, contrairement à nos attentes, cette configuration a entraîné une **baisse des performances** avec un **best fitness de 0.23**. Nous avons remarqué que l'ajout de `mosaic` et `mixup` compliquait l'apprentissage, probablement en raison de la génération d'assets entraînant une mauvaise généralisation du modèle.

### 3. Tuning Automatique avec `model.tune` 🎯

Nous avons tenté d'affiner encore davantage les performances du modèle en utilisant le tuning automatique des hyperparamètres via `model.tune`. Cette technique ajuste dynamiquement les paramètres en fonction des performances obtenues au fil des itérations.

```python
model.tune(
    data=os.path.abspath("./config.yaml"),
    epochs=30,
    iterations=150,
    optimizer="AdamW",
    imgsz=640,
)
```

Toutefois, cette approche n'a pas apporté les améliorations escomptées, et nous avons obtenu un **best fitness autour de 0.22**. Il semble que le tuning automatique peine à trouver des combinaisons d'hyperparamètres pertinentes pour notre dataset spécifique.

---

## Observations et Conclusions 🔎

- **La configuration de base a donné les meilleurs résultats (0.30)**, confirmant que les hyperparamètres standards sont plus adaptés à notre tâche.
- **L'ajout de data augmentation a dégradé les performances**, probablement à cause de la complexité accrue des transformations appliquées aux images.
- **L'entraînement avec tuning automatique n'a pas permis de surpasser la configuration initiale**, indiquant une possible inadéquation entre notre dataset et les paramètres ajustés.
- **Nos résultats restent insuffisants pour une application réelle**, le best fitness plafonnant à 0.30.

---

## Pistes d'Amélioration 🚀

Pour améliorer les performances du modèle, nous proposons plusieurs axes d'exploration :

- **Tester d'autres architectures YOLO** comme YOLOv11s, m, l, x , qui pourraient offrir de meilleures résultats.
- **Ajuster la taille des images** (`imgsz: 768` ou `896`) afin d'améliorer la résolution des objets détectés et éviter les pertes d'information.
- **Rééquilibrer les classes du dataset** pour s'assurer que la distribution des données ne biaise pas l'entraînement du modèle.
- **Expérimenter avec des optimisateurs différents**, comme `SGD` avec un `momentum` ajusté, afin d'améliorer la convergence.
- **Essayer des stratégies de data augmentation plus légères**, telles que le `flip`, la `rotation` et l'ajout de bruit, plutôt que des techniques trop agressives.

---

## Lien vers les Modèles 🔗

- **Meilleur modèle atteint** : [Voir sur Picsellia](https://app.picsellia.com/0192f6db-86b6-784c-80e6-163debb242d5/model/01936428-4088-7381-8513-9813994a8a7b/version/0194e6c4-fdee-7c35-940c-06fc97804a6a)
- **Dernier modèle utilisé en inférence** : [Voir sur Picsellia](https://app.picsellia.com/0192f6db-86b6-784c-80e6-163debb242d5/model/01936428-4088-7381-8513-9813994a8a7b/version/0194ea70-a358-7692-bb96-440be99705a9)

---

## Conclusion 🎯

Malgré de nombreuses expérimentations et ajustements, notre modèle **n’a pas atteint une performance satisfaisante**. Nos tests indiquent que **les hyperparamètres de base restent les plus efficaces** comparés aux configurations plus complexes. Toutefois, des pistes d’amélioration subsistent, notamment en ajustant **l’architecture du modèle, la taille des images et les stratégies de prétraitement**.

Nous recommandons d'explorer ces directions pour espérer une meilleure convergence et une amélioration significative des performances du modèle. 🚀
