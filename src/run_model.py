from torch.optim import SGD
from pyjet.models import SLModel
from pyjet.callbacks import ModelCheckpoint, Plotter
from pyjet.data import NpDataset

from models import SimpleModel
import utils

MODEL = SimpleModel
RUN_ID = "simple"


def train_model(model: SLModel,
                trainset: NpDataset,
                valset: NpDataset,
                epochs=25,
                batch_size=32):

    # Create the generators
    traingen = trainset.flow(batch_size=batch_size, shuffle=True)
    valgen = valset.flow(batch_size=batch_size, shuffle=False)

    # Create the callbacks
    callbacks = [
        ModelCheckpoint(
            utils.get_cv_path(RUN_ID), "loss", verbose=1, save_best_only=True),
        Plotter(
            "loss",
            scale='log',
            plot_during_train=False,
            save_to_file=utils.get_plot_path(RUN_ID),
            block_on_end=False),
        Plotter(
            "accuracy",
            scale='linear',
            plot_during_train=False,
            save_to_file=utils.get_plot_path(RUN_ID + "_acc"),
            block_on_end=False)
    ]

    # Create the optiizer
    optimizer = SGD(
        [param for param in model.parameters() if param.requires_grad],
        momentum=0.9,
        nesterov=True)

    # Train the model
    model.fit_generator(
        traingen,
        traingen.steps_per_epoch,
        epochs=epochs,
        optimizer=optimizer,
        validation_generator=valgen,
        validation_steps=valgen.steps_per_epoch,
        metrics=["accuracy"],
        callbacks=callbacks,
        verbose=1)

    return model


def train(data):
    data.
