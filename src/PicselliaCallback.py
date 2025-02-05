from picsellia.types.enums import LogType

class PicselliaCallback:
    def __init__(self, experiment):
        self.experiment = experiment

    def on_train_epoch_end(self, trainer):
        # Loguer chaque métrique une par une
        for key, value in trainer.metrics.items():
            try:
                self.experiment.log(key, value, LogType.LINE)
            except Exception as e:
                print(f"Erreur lors du logging de {key}: {e}")
