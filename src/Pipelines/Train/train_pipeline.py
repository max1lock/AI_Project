import os
import yaml
from ultralytics import YOLO
from src.Picsellia.PicselliaCallback import PicselliaCallback
from picsellia.types.enums import InferenceType, Framework
from src.Pipelines.Train import data_preparation
from src.Picsellia.PicselliaConfig import (
    get_client,
    get_project,
    get_model_object,
)
from config import settings


def get_new_experiment_name(base_name, experiments):
    """
    Parcourt la liste des expériences pour extraire le dernier suffixe numérique associé
    au nom de base (par exemple 'experiment') et retourne un nouveau nom avec le suffixe incrémenté.

    Si aucune expérience ne contient de suffixe, le nom de base est retourné.

    Args:
        base_name (str): Nom de base de l'expérience (ex. "experiment").
        experiments (list): Liste d'objets expérience ayant un attribut 'name'.

    Returns:
        str: Nouveau nom pour l'expérience.
    """
    last_number = 0
    for exp in experiments:
        name = exp.name
        if name == base_name:
            last_number = max(last_number, 1)
        elif name.startswith(f"{base_name}_"):
            try:
                number = int(name.split("_")[-1])
                last_number = max(last_number, number)
            except ValueError:
                continue
    return f"{base_name}_{last_number + 1}" if last_number else base_name


def train_model() -> None:
    """
    Pipeline d'entraînement du modèle YOLO.
    :return:
    """
    api_key = settings.api_key
    workspace_name = settings.workspace_name
    dataset_uuid = settings.dataset_uuid
    project_name = settings.groupe_name
    model_name = settings.groupe_name

    client = get_client(api_key, workspace_name)
    project = get_project(client, project_name)
    model_obj = get_model_object(client, model_name)

    base_name = "experiment"
    existing_experiments = project.list_experiments()
    new_experiment_name = get_new_experiment_name(
        base_name, existing_experiments
    )
    experiment = project.create_experiment(name=new_experiment_name)
    experiment.attach_dataset(
        dataset_uuid, client.get_dataset_version_by_id(dataset_uuid)
    )
    print(f"Nouvelle expérimentation créée : {experiment.name}")

    dataset = data_preparation.download_images(client, dataset_uuid)
    data_preparation.export_and_extract_annotations(dataset)
    images_dir = data_preparation.structure_dataset()
    config_yaml_path = "./config.yaml"
    data_preparation.generate_yaml_file(config_yaml_path, images_dir)

    with open("hyperparameters.yaml", "r") as file:
        hyperparameters_config = yaml.safe_load(file)

    model = YOLO("yolo11n.pt")
    model.to(device="cuda:0")

    picsellia_callback = PicselliaCallback(experiment)
    model.add_callback(
        "on_train_epoch_end", picsellia_callback.on_train_epoch_end
    )

    model.train(
        data=os.path.abspath(config_yaml_path),
        project=new_experiment_name,
        name="train_results",
        device="0",
        **hyperparameters_config,
    )

    predictions = model.predict(source="./datasets/structured/images/test")

    class_names = model.names

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

    model_save_path = f"./{new_experiment_name}/train_results/weights/best.pt"
    print(f"Modèle entraîné sauvegardé dans : {model_save_path}")

    inference_type = InferenceType.OBJECT_DETECTION
    framework = Framework.PYTORCH
    model_version = model_obj.create_version(
        type=inference_type, framework=framework
    )
    model_version.store(name="best", path=model_save_path)
    experiment.attach_model_version(model_version)

    print(
        "Nouvelle version du modèle 'Groupe_6' créée et uploadée avec succès."
    )
