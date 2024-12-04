import os
from picsellia import Client
from picsellia.types.enums import AnnotationFileType

# Configuration
api_key = "6a0008897629d989cf385e35ff3c60e45b355584"
workspace_name = "Picsalex-MLOps"
dataset_name = "⭐️ cnam_product_2024"
dataset_uuid = "0193688e-aa8f-7cbe-9396-bec740a262d0"

# Connexion au client Picsellia
client = Client(api_token=api_key, organization_name=workspace_name)

# Récupération du dataset
dataset = client.get_dataset_version_by_id(dataset_uuid)
# Téléchargement du dataset
dataset.list_assets().download("./datasets")

# Dossier où enregistrer les annotations
output_dir = "./datasets/annotations"
os.makedirs(output_dir, exist_ok=True)

# Chemin pour le fichier ZIP des annotations
annotations_zip_path = os.path.join(output_dir, "annotations.zip")

# Exporter les annotations au format YOLO
print("Exportation des annotations au format YOLO...")
dataset.export_annotation_file(AnnotationFileType.YOLO, annotations_zip_path)

print(f"Annotations téléchargées dans : {annotations_zip_path}")

# Extraction des fichiers ZIP
import zipfile

print("Extraction des annotations...")
with zipfile.ZipFile(annotations_zip_path, "r") as zip_ref:
    zip_ref.extractall(output_dir)

print(f"Annotations extraites dans : {output_dir}")
