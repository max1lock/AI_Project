import os
import zipfile
import shutil
import random
from glob import glob
import yaml
from picsellia.types.enums import AnnotationFileType


def download_images(client, dataset_uuid, target_folder="./datasets/images"):
    """
    Télécharge les images du dataset s'il n'existe pas déjà.
    """
    dataset = client.get_dataset_version_by_id(dataset_uuid)
    os.makedirs(target_folder, exist_ok=True)
    if not os.listdir(target_folder):
        print("Téléchargement des images...")
        dataset.list_assets().download(target_path=target_folder, use_id=True)
    return dataset


def export_and_extract_annotations(
    dataset,
    annotations_zip_path="./datasets/annotations.zip",
    output_dir="./datasets/annotations",
):
    """
    Exporte les annotations au format YOLO et extrait le zip.
    """
    os.makedirs(output_dir, exist_ok=True)
    if not os.listdir(output_dir):
        print("Exportation des annotations au format YOLO...")
        dataset.export_annotation_file(
            annotation_file_type=AnnotationFileType.YOLO,
            target_path=annotations_zip_path,
            use_id=True,
        )

        print("Extraction des annotations...")
        with zipfile.ZipFile(
            r".\datasets\annotations.zip\0192f6db-86b6-784c-80e6-163debb242d5\annotations\0193688e-aa8f-7cbe-9396-bec740a262d0_annotations.zip",
            "r",
        ) as zip_ref:
            zip_ref.extractall(output_dir)
        print(f"Annotations extraites dans : {output_dir}")
    return output_dir


def structure_dataset(
    images_source="./datasets/images",
    annotations_source="./datasets/annotations",
    output_folder="./datasets/structured",
    split_ratios={"train": 0.6, "val": 0.2, "test": 0.2},
):
    """
    Réorganise les images et les annotations dans des sous-dossiers pour l'entraînement.
    """
    images_dir = f"{output_folder}/images"
    labels_dir = f"{output_folder}/labels"
    splits = ["train", "val", "test"]

    # Création des dossiers pour chaque split
    for split in splits:
        os.makedirs(f"{images_dir}/{split}", exist_ok=True)
        os.makedirs(f"{labels_dir}/{split}", exist_ok=True)

    # Récupérer les fichiers
    all_images = sorted(glob(f"{images_source}/*.jpg"))
    all_labels = sorted(glob(f"{annotations_source}/*.txt"))

    # Attention : assurez-vous que l'association image/annotation se fait correctement
    data_pairs = list(zip(all_images, all_labels))
    random.seed(42)
    random.shuffle(data_pairs)

    nb_total = len(data_pairs)
    train_end = int(nb_total * split_ratios["train"])
    val_end = train_end + int(nb_total * split_ratios["val"])

    splits_data = {
        "train": data_pairs[:train_end],
        "val": data_pairs[train_end:val_end],
        "test": data_pairs[val_end:],
    }

    # Copier les fichiers dans les répertoires correspondants
    for split, pairs in splits_data.items():
        for image_path, label_path in pairs:
            shutil.copy(image_path, f"{images_dir}/{split}")
            shutil.copy(label_path, f"{labels_dir}/{split}")

    print(f"Dataset structuré dans : {output_folder}")
    return (
        images_dir  # On retournera le dossier des images pour la config YAML
    )


def generate_yaml_file(yaml_path, images_dir):
    """
    Génère le fichier YAML de configuration pour YOLO.
    """
    data = {
        "train": f"{images_dir}/train",
        "val": f"{images_dir}/val",
        "test": f"{images_dir}/test",
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
    with open(yaml_path, "w") as f:
        yaml.dump(data, f)
    print(f"Fichier de config YAML généré : {yaml_path}")
