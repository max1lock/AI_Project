import os
from fileinput import filename

import cv2
import argparse
from picsellia import Client, ModelVersion
from scipy.misc import ascent
from ultralytics import YOLO
import re


def extract_model_number(version):
    # On suppose que str(version) renvoie quelque chose du type "Version Groupe_6-15 of Model '15' ..."
    match = re.search(r"Model '(\d+)'", str(version))
    if match:
        return int(match.group(1))
    # En cas d'absence de numéro, on renvoie -1 pour être sûr de le ne pas prendre en compte
    return -1


def load_model_from_picsellia(
    api_key: str, workspace_name: str, model_name: str
) -> str:
    """
    Récupère la version du modèle depuis Picsellia et télécharge le fichier best.pt.

    Args:
        api_key (str): Clé API de Picsellia.
        workspace_name (str): Nom de l'organisation sur Picsellia.
        model_name (str): Nom du modèle à récupérer.

    Returns:
        str: Chemin local du fichier best.pt.
    """
    client = Client(api_token=api_key, organization_name=workspace_name)

    model_name = "Groupe_6"
    model = client.get_model(name=model_name)

    model_versions = model.list_versions()
    last_version = max(model_versions, key=extract_model_number)

    path_to_download = "./models"
    client.get_model_version_by_id(last_version.id).get_file(
        name="best"
    ).download(target_path=path_to_download, force_replace=True)

    model_path = path_to_download + "/best.pt"
    print(f"✅ Modèle téléchargé : {model_path}")

    return model_path


def run_inference(model_path: str, mode: str, source: str = None):
    """
    Effectue une inférence avec le modèle YOLO sur une image, une vidéo ou une webcam.

    Args:
        model_path (str): Chemin vers le modèle best.pt.
        mode (str): Mode d'inférence (IMAGE, VIDEO, WEBCAM).
        source (str, optional): Chemin du fichier pour IMAGE ou VIDEO. Non requis pour WEBCAM.
    """
    model = YOLO(model_path)

    if mode == "IMAGE":
        if source is None:
            raise ValueError("❌ Chemin d'image requis pour le mode IMAGE.")
        results = model(source)
        # Comme results est une liste, on parcourt chaque résultat et on appelle show()
        for r in results:
            r.show()  # Affiche l'image annotée
        print("✅ Inférence sur IMAGE terminée.")

    elif mode == "VIDEO":
        if source is None:
            raise ValueError("❌ Chemin de vidéo requis pour le mode VIDEO.")
        # Utilisation de cv2.VideoCapture pour lire le fichier vidéo
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError("❌ Impossible d'ouvrir la vidéo.")
        print(
            "🎥 Inférence sur VIDEO en cours... (Appuyez sur 'q' pour quitter)"
        )
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Appliquer le modèle sur la frame
            results = model(frame)
            # On utilise plot() sur le premier résultat pour obtenir l'image annotée
            annotated_frame = results[0].plot()
            cv2.imshow("YOLO Video Inference", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Inférence sur VIDEO terminée.")

    elif mode == "WEBCAM":
        cap = cv2.VideoCapture(0)
        print(
            "🎥 Inférence en temps réel via WEBCAM... (Appuyez sur 'q' pour quitter)"
        )
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)
            frame_rendered = results[0].plot()
            cv2.imshow("YOLO Inference", frame_rendered)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Inférence WEBCAM terminée.")

    else:
        raise ValueError(
            "❌ Mode invalide. Choisissez IMAGE, VIDEO ou WEBCAM."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline d'inférence avec YOLO et Picsellia."
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["IMAGE", "VIDEO", "WEBCAM"],
        help="Mode d'inférence.",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=False,
        help="Chemin du fichier (requis pour IMAGE et VIDEO).",
    )

    args = parser.parse_args()

    # Connexion et récupération du modèle
    API_KEY = "6a0008897629d989cf385e35ff3c60e45b355584"
    WORKSPACE_NAME = "Picsalex-MLOps"
    MODEL_NAME = "Groupe_6"

    model_path = load_model_from_picsellia(API_KEY, WORKSPACE_NAME, MODEL_NAME)

    # Exécuter l'inférence
    run_inference(model_path, args.mode, args.source)
