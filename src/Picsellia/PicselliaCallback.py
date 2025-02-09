from picsellia.types.enums import LogType


class PicselliaCallback:
    def __init__(self, experiment):
        self.experiment = experiment

    def on_train_epoch_end(self, trainer):
        try:
            for name, value in trainer.metrics.items():
                self.experiment.log(name, [value], LogType.LINE)
            self.experiment.log(
                "Epoch duration", [trainer.epoch_time], LogType.BAR
            )
            self.experiment.log("Fitness", [trainer.fitness], LogType.LINE)
            if trainer.best_fitness is not None:
                self.experiment.log(
                    "Best fitness", trainer.best_fitness, LogType.VALUE
                )
            for index, loss_name in enumerate(trainer.loss_names):
                self.experiment.log(
                    loss_name, [trainer.loss_items[index].item()], LogType.LINE
                )
        except Exception as e:
            print(f"Erreur lors du logging de {name}: {e}")
