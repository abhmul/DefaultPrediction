import logging

from torch.optim import SGD
from pyjet.models import SLModel
from pyjet.callbacks import ModelCheckpoint, Plotter
from pyjet.data import NpDataset

from sklearn.metrics import roc_auc_score

from models import SimpleModel
import utils
from data_utils import HomeCreditData

MODEL = SimpleModel
RUN_ID = "simple"
SEED = 42
utils.set_random_seed(SEED)
SPLIT_SEED = utils.get_random_seed()


# TODO: Add ROC AUC stateful metric to pyjet so we don't need the validate
# function and can plot the roc-auc over time
def train_model(model: SLModel,
                trainset: NpDataset,
                valset: NpDataset,
                epochs=25,
                batch_size=32):

    # Create the generators
    traingen = trainset.flow(
        batch_size=batch_size, shuffle=True, seed=utils.get_random_seed())
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
    logs = model.fit_generator(
        traingen,
        traingen.steps_per_epoch,
        epochs=epochs,
        optimizer=optimizer,
        validation_generator=valgen,
        validation_steps=valgen.steps_per_epoch,
        metrics=["accuracy"],
        callbacks=callbacks,
        verbose=1)

    return logs


def validate_model(model: SLModel, val_data: NpDataset, batch_size=32):
    valgen = val_data.flow(batch_size=batch_size, shuffle=False)
    val_preds = model.predict_generator(valgen, valgen.steps_per_epoch)
    return roc_auc_score(val_data.y, val_preds)


def train(data: HomeCreditData):
    train_data = data.load_train()
    model = MODEL(input_size=train_data.x.shape[1])
    train_data, val_data = train_data.validation_split(
        split=0.1, shuffle=True, stratified=True, seed=SPLIT_SEED)
    train_model(model, train_data, val_data)
    # Load the model and score it
    model = model.load_state(utils.get_model_path(RUN_ID))
    score = validate_model(model, val_data)
    logging.info("ROC AUC score of best model is {}".format(score))
    return model
