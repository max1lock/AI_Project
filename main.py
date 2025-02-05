import os
import shutil
import random
from picsellia import Client, ModelVersion
from picsellia.types.enums import AnnotationFileType, Framework
from glob import glob
import yaml
import zipfile
from ultralytics import YOLO
import torch

from src.PicselliaCallback import PicselliaCallback
from picsellia.types.enums import InferenceType


def main():
    # Configuration
    api_key = "6a0008897629d989cf385e35ff3c60e45b355584"
    workspace_name = "Picsalex-MLOps"
    dataset_name = "⭐️ cnam_product_2024"
    dataset_uuid = "0193688e-aa8f-7cbe-9396-bec740a262d0"

    # Connexion au client Picsellia
    client = Client(api_token=api_key, organization_name=workspace_name)

    project = client.get_project(project_name="Groupe_6")

    # Récupérer l'objet modèle existant "Groupe_6"
    model_name = "Groupe_6"
    model_obj = client.get_model(name=model_name)

    experiment_name = "experiment"
    existing_experiments = project.list_experiments()
    # Increment the experiment name
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

        # Suppression du fichier ZIP
        # os.remove(r".\datasets\annotations.zip")

    # ---------- PARTIE 2 : Structurer les données pour Ultralytics YOLO ----------

    # Chemin pour le fichier ZIP des annotations

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
    def copy_files(pairs, dest_image_dir, dest_label_dir):
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

    def generate_yaml_file(output_path):
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

    # ---------- PARTIE 3 : Entraînement du modèle avec YOLO v11 ----------

    # Charger le modèle YOLOv11
    model = YOLO("yolo11n.pt")

    # use cuda gpu
    model.to(device="cuda:0")

    # Attacher le callback de Picsellia pour envoyer des logs après chaque époque
    picsellia_callback = PicselliaCallback(experiment)
    model.add_callback(
        "on_train_epoch_end", picsellia_callback.on_train_epoch_end
    )

    # Configurer et lancer l'entraînement
    model.train(
        data=os.path.abspath(
            "./config.yaml"
        ),  # Chemin vers votre fichier config.yaml,
        epochs=10,
        batch=32,
        optimizer="Adam",
        lr0=0.001,
        imgsz=640,
        project=experiment_name,
        name="train_results",
        device="0",
    )

    predictions = model.predict(
        "./datasets/structured/images/test"
    )  # Prédiction sur le jeu de test

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

    # Assuming 'predictions' is a list of predictions for each image in the test set
    for prediction in predictions:
        # Assuming each prediction contains boxes, classes, and confidence scores
        for box, class_id, score in zip(
            prediction.boxes.xyxy, prediction.boxes.cls, prediction.boxes.conf
        ):
            # Get the image file name for the current prediction
            image_file_name = os.path.basename(prediction.path)
            asset_id = image_file_name.split(".")[0]

            # Find the corresponding asset in your Picsellia dataset
            asset = dataset.find_asset(id=asset_id)

            # Assuming your dataset has labels with the same names as your model's class IDs
            label_name = class_names[int(class_id)]
            label = dataset.get_label(name=label_name)

            # Add the evaluation for the current bounding box
            experiment.add_evaluation(
                asset,
                rectangles=[
                    (
                        int(box[0].item()),  # xmin
                        int(box[1].item()),  # ymin
                        int(box[2].item()),  # xmax
                        int(box[3].item()),  # ymax
                        label,  # l'objet Label Picsellia
                        score.item(),  # score de confiance
                    )
                ],
            )

    job = experiment.compute_evaluations_metrics(
        InferenceType.OBJECT_DETECTION
    )
    job.wait_for_done()

    # Sauvegarde du model
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

    # Attacher le model à l'expérimentation
    experiment.attach_model_version(model_version)

    print(
        "Nouvelle version du modèle 'Groupe_6' créée et modèle uploadé avec succès."
    )


if __name__ == "__main__":
    main()
