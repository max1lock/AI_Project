import os
import shutil
import random
from picsellia import Client, ModelVersion
from picsellia.types.enums import AnnotationFileType, Framework
from glob import glob
import yaml
import zipfile

from torch import optim
from torch.optim.lr_scheduler import StepLR
from ultralytics import YOLO
import torch
from src.PicselliaCallback import PicselliaCallback
from picsellia.types.enums import InferenceType


def main() -> None:
    """
    Main function to execute the full pipeline of downloading dataset,
    structuring data, training a YOLO model, and evaluating the results.
    """
    # Configuration
    api_key: str = "6a0008897629d989cf385e35ff3c60e45b355584"
    workspace_name: str = "Picsalex-MLOps"
    dataset_name: str = "⭐️ cnam_product_2024"
    dataset_uuid: str = "0193688e-aa8f-7cbe-9396-bec740a262d0"

    # Connexion au client Picsellia
    client: Client = Client(
        api_token=api_key, organization_name=workspace_name
    )

    project = client.get_project(project_name="Groupe_6")

    # Récupérer l'objet modèle existant "Groupe_6"
    model_name: str = "Groupe_6"
    model_obj: ModelVersion = client.get_model(name=model_name)

    experiment_name: str = "experiment"
    existing_experiments = project.list_experiments()

    # Increment the experiment name if necessary
    experiment_name = (
        f"{experiment_name}_{len(existing_experiments) + 1}"
        if experiment_name in [exp.name for exp in existing_experiments]
        else experiment_name
    )

    experiment = project.create_experiment(name=experiment_name)

    # Attacher le dataset à l'expérimentation
    experiment.attach_dataset(
        dataset_uuid, client.get_dataset_version_by_id(dataset_uuid)
    )
    print(f"Nouvelle expérimentation créée : {experiment.name}")

    # ---------- PARTIE 1 : Téléchargement du dataset ----------
    dataset = client.get_dataset_version_by_id(dataset_uuid)

    # Si le dossier datasets ne contient aucun fichier
    if not os.path.exists("./datasets"):
        os.makedirs("./datasets")
    if not os.listdir("./datasets"):
        # Récupération du dataset
        if not os.path.exists("./datasets/images"):
            os.makedirs("./datasets/images")
        if not os.listdir("./datasets/images"):

            # Téléchargement du dataset
            dataset.list_assets().download(
                target_path="./datasets/images", use_id=True
            )

        # Dossier où enregistrer les annotations
        output_dir = "./datasets/annotations"
        os.makedirs(output_dir, exist_ok=True)

        # Chemin pour le fichier ZIP des annotations
        annotations_zip_path = "./datasets/annotations.zip"

        # Exporter les annotations au format YOLO
        print("Exportation des annotations au format YOLO...")
        dataset.export_annotation_file(
            annotation_file_type=AnnotationFileType.YOLO,
            target_path="./datasets/annotations.zip",
            use_id=True,
        )

        print(f"Annotations téléchargées dans : {annotations_zip_path}")

        # Extraction des fichiers ZIP
        print("Extraction des annotations...")
        with zipfile.ZipFile(
            r".\datasets\annotations.zip\0192f6db-86b6-784c-80e6-163debb242d5\annotations\0193688e-aa8f-7cbe-9396-bec740a262d0_annotations.zip",
            "r",
        ) as zip_ref:
            zip_ref.extractall(output_dir)

        print(f"Annotations extraites dans : {output_dir}")

    # ---------- PARTIE 2 : Structurer les données pour Ultralytics YOLO ----------
    output_dir = "./datasets/structured"
    images_dir = f"{output_dir}/images"
    labels_dir = f"{output_dir}/labels"
    train_dir = "train"
    val_dir = "val"
    test_dir = "test"
    random.seed(42)  # Fixer la seed pour la reproductibilité
    split_ratios = {"train": 0.6, "val": 0.2, "test": 0.2}

    # Créer les répertoires de sortie
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(f"{images_dir}/{train_dir}", exist_ok=True)
    os.makedirs(f"{images_dir}/{val_dir}", exist_ok=True)
    os.makedirs(f"{images_dir}/{test_dir}", exist_ok=True)

    os.makedirs(f"{labels_dir}/{train_dir}", exist_ok=True)
    os.makedirs(f"{labels_dir}/{val_dir}", exist_ok=True)
    os.makedirs(f"{labels_dir}/{test_dir}", exist_ok=True)

    # Liste des images et labels
    all_images = glob("./datasets/images/*.jpg")
    all_labels = glob("./datasets/annotations/*.txt")

    # Associer images et labels
    data_pairs = list(zip(all_images, all_labels))
    random.shuffle(data_pairs)

    # Calcul des indices pour chaque split
    train_split = int(len(data_pairs) * split_ratios["train"])
    val_split = train_split + int(len(data_pairs) * split_ratios["val"])

    # Répartition
    train_pairs = data_pairs[:train_split]
    val_pairs = data_pairs[train_split:val_split]
    test_pairs = data_pairs[val_split:]

    # Copier les fichiers dans les répertoires correspondants
    def copy_files(
        pairs: list, dest_image_dir: str, dest_label_dir: str
    ) -> None:
        """
        Copie les fichiers image et label dans les répertoires correspondants.

        Args:
            pairs (list): Liste des tuples (image, label).
            dest_image_dir (str): Répertoire de destination des images.
            dest_label_dir (str): Répertoire de destination des labels.
        """
        for image, label in pairs:
            shutil.copy(image, dest_image_dir)
            shutil.copy(label, dest_label_dir)

    copy_files(
        train_pairs, f"{images_dir}/{train_dir}", f"{labels_dir}/{train_dir}"
    )
    copy_files(val_pairs, f"{images_dir}/{val_dir}", f"{labels_dir}/{val_dir}")
    copy_files(
        test_pairs, f"{images_dir}/{test_dir}", f"{labels_dir}/{test_dir}"
    )

    def generate_yaml_file(output_path: str) -> None:
        """
        Génère le fichier YAML de configuration pour l'entraînement de YOLO.

        Args:
            output_path (str): Chemin de sortie pour le fichier YAML.
        """
        data = {
            "train": f"{images_dir}/{train_dir}",
            "val": f"{images_dir}/{val_dir}",
            "test": f"{images_dir}/{test_dir}",
            "nc": 10,  # Nombre de classes
            "names": [
                "mikado",
                "kinder_pingui",
                "kinder_country",
                "kinder_tronky",
                "tic_tac",
                "sucette",
                "capsule",
                "pepito",
                "bouteille_plastique",
                "canette",
            ],
        }
        with open(output_path, "w") as yaml_file:
            yaml.dump(data, yaml_file)
        print(f"Fichier config.yaml généré : {output_path}")

    generate_yaml_file("./config.yaml")

    # Charger la configuration de data augmentation
    with open("data_augmentation.yaml", "r") as file:
        data_aug_config = yaml.safe_load(file)["augmentations"]

    # ---------- PARTIE 3 : Entraînement du modèle avec YOLO v11 ----------
    model = YOLO("yolo11s.pt")

    # use cuda gpu
    model.to(device="cuda:0")

    # Attacher le callback de Picsellia pour envoyer des logs après chaque époque
    picsellia_callback = PicselliaCallback(experiment)
    model.add_callback(
        "on_train_epoch_end", picsellia_callback.on_train_epoch_end
    )

    model.train(
        data=os.path.abspath("./config.yaml"),
        epochs=100,  # Nombre d'époques
        batch=16,  # Taille du batch
        optimizer="AdamW",  # Optimiseur
        lr0=0.001,  # Taux d'apprentissage
        imgsz=640,  # Taille des images
        project=experiment_name,  # Nom du projet pour stocker les résultats
        name="train_results",  # Nom du répertoire pour les résultats
        device="cuda:0",  # Utilisation du GPU
        patience=10,  # Arrêt anticipé après 10 époques sans amélioration
        workers=8,  # Nombre de workers pour le chargement des données
        exist_ok=True,  # Ignorer les erreurs si le répertoire existe
        **data_aug_config,  # Injection des paramètres de data augmentation
    )

    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    model.train(
        data=os.path.abspath("./config.yaml"),
        epochs=100,
        batch=8,
        optimizer="SGD",
        lr0=0.0001,
        imgsz=640,
        project=experiment_name,
        name="train_results",
        device="cuda:0",
        # patience=10,
        workers=8,
        exist_ok=True,
        seed=42,
        close_mosaic=0,
    )

    predictions = model.predict("./datasets/structured/images/test")

    class_names = [
        "mikado",
        "kinder_pingui",
        "kinder_country",
        "kinder_tronky",
        "tic_tac",
        "sucette",
        "capsule",
        "pepito",
        "bouteille_plastique",
        "canette",
    ]

    for prediction in predictions:
        for box, class_id, score in zip(
            prediction.boxes.xyxy, prediction.boxes.cls, prediction.boxes.conf
        ):
            image_file_name = os.path.basename(prediction.path)
            asset_id = image_file_name.split(".")[0]

            asset = dataset.find_asset(id=asset_id)
            label_name = class_names[int(class_id)]
            label = dataset.get_label(name=label_name)

            experiment.add_evaluation(
                asset,
                rectangles=[
                    (
                        int(box[0].item()),
                        int(box[1].item()),
                        int(box[2].item()),
                        int(box[3].item()),
                        label,
                        score.item(),
                    )
                ],
            )

    job = experiment.compute_evaluations_metrics(
        InferenceType.OBJECT_DETECTION
    )
    job.wait_for_done()

    # Sauvegarde du modèle
    model_save_path = "./trained_model.pt"
    model.save(model_save_path)
    print(f"Modèle entraîné sauvegardé dans : {model_save_path}")

    # Envoi du modèle sur Picsellia
    inference_type = InferenceType.OBJECT_DETECTION
    framework = Framework.PYTORCH

    model_version = model_obj.create_version(
        type=inference_type, framework=framework
    )

    model_version.store(name="best", path=model_save_path)
    experiment.attach_model_version(model_version)

    print(
        "Nouvelle version du modèle 'Groupe_6' créée et modèle uploadé avec succès."
    )


if __name__ == "__main__":
    main()
