import os
import shutil
import random
from picsellia import Client
from picsellia.types.enums import AnnotationFileType
from glob import glob
import yaml
import zipfile
from ultralytics import YOLO
import torch

# Configuration
api_key = "6a0008897629d989cf385e35ff3c60e45b355584"
workspace_name = "Picsalex-MLOps"
dataset_name = "⭐️ cnam_product_2024"
dataset_uuid = "0193688e-aa8f-7cbe-9396-bec740a262d0"

# Connexion au client Picsellia
client = Client(api_token=api_key, organization_name=workspace_name)

project = client.get_project(project_name="Groupe_6")

experiment_name = "experiment_1"
existing_experiments = project.list_experiments()
for exp in existing_experiments:
    if exp.name == experiment_name:
        print(
            f"Une expérimentation avec le nom '{experiment_name}' existe déjà. Suppression en cours..."
        )
        exp.delete()
        print(f"Expérimentation '{experiment_name}' supprimée.")

experiment = project.create_experiment(name=experiment_name)
# Attacher le dataset à l'expérimentation
experiment.attach_dataset(
    dataset_uuid, client.get_dataset_version_by_id(dataset_uuid)
)
print(f"Nouvelle expérimentation créée : {experiment.name}")
# ---------- PARTIE 1 : Téléchargement du dataset ----------

# Si le dossier datasets ne contient aucun fichier
if not os.path.exists("./datasets"):
    os.makedirs("./datasets")
if not os.listdir("./datasets"):

    dataset = client.get_dataset_version_by_id(dataset_uuid)
    # Récupération du dataset
    if not os.path.exists("./datasets/images"):
        os.makedirs("./datasets/images")
    if not os.listdir("./datasets/images"):
        # Téléchargement du dataset
        dataset.list_assets().download("./datasets/images")

    # Dossier où enregistrer les annotations
    output_dir = "./datasets/annotations"
    os.makedirs(output_dir, exist_ok=True)

    # Chemin pour le fichier ZIP des annotations
    annotations_zip_path = "./datasets/annotations.zip"

    # Exporter les annotations au format YOLO
    print("Exportation des annotations au format YOLO...")
    dataset.export_annotation_file(
        AnnotationFileType.YOLO, "./datasets/annotations.zip"
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
copy_files(test_pairs, f"{images_dir}/{test_dir}", f"{labels_dir}/{test_dir}")


def generate_yaml_file(output_path):
    data = {
        "train": f"{images_dir}/{train_dir}",
        "val": f"{images_dir}/{val_dir}",
        "test": f"{images_dir}/{test_dir}",
        "nc": 10,  # Nombre de classes
        "names": [
            "Canettes",
            "Bouteilles en plastique",
            "Pepito",
            "Kinder Country",
            "Kinder Tronky",
            "Kinder Pinguy",
            "Tic-Tac",
            "Sucette",
            "Capsule",
            "Mikado",
        ],
    }
    with open(output_path, "w") as yaml_file:
        yaml.dump(data, yaml_file)
    print(f"Fichier config.yaml généré : {output_path}")


generate_yaml_file("./config.yaml")

# ---------- PARTIE 3 : Entraînement du modèle avec YOLO v11 ----------

torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Charger le modèle YOLOv11
model = YOLO("yolo11n.pt", task="detect")

model.to(device=device)

# Tune hyperparameters for 30 epochs
if not (os.path.exists("./runs")):
    model.tune(
        data=r"C:\Users\Alexandre\PycharmProjects\AI_Project\config.yaml",
        epochs=30,
        iterations=1,
        optimizer="AdamW",
        plots=False,
        save=False,
        val=False,
    )

# Train using config.yaml and hyperparameters
# Configurer et lancer l'entraînement
model.train(
    data=r"C:\Users\Alexandre\PycharmProjects\AI_Project\config.yaml",
    hyp=r"C:\Users\Alexandre\PycharmProjects\AI_Project\runs\detect\tune\best_hyperparameters.yaml",
    name="experiment_1",
    project=r"C:\Users\Alexandre\PycharmProjects\AI_Project\results",
)
