import logging
import argparse

import torch.optim as optim
from pyjet.models import SLModel
from pyjet.callbacks import ModelCheckpoint, Plotter
from pyjet.data import NpDataset

from sklearn.metrics import roc_auc_score

from models import SimpleModel
import utils
from data_utils import HomeCreditData

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--data', default='../input/', help="Path to the data")
parser.add_argument(
    '--train', action='store_true', help="Runs the script in train mode")
parser.add_argument(
    '--test', action='store_true', help="Runs the script in test mode")

MODEL = SimpleModel
RUN_ID = "simple"
SEED = 42
utils.set_random_seed(SEED)
SPLIT_SEED = utils.get_random_seed()


# TODO: Add ROC AUC stateful metric to pyjet so we don't need the validate
# function and can plot the roc-auc over time
# TODO: add more logging to these functions s we can see what's going on
def train_model(model: SLModel,
                trainset: NpDataset,
                valset: NpDataset,
                epochs=5,
                batch_size=32):

    # Create the generators
    logging.info("Training model for {} epochs and {} batch size".format(
        epochs, batch_size))
    logging.info("Flowing the train and validation sets")
    traingen = trainset.flow(
        batch_size=batch_size, shuffle=True, seed=utils.get_random_seed())
    valgen = valset.flow(batch_size=batch_size, shuffle=False)

    # Create the callbacks
    logging.info("Creating the callbacks")
    callbacks = [
        ModelCheckpoint(
            utils.get_model_path(RUN_ID),
            "val_loss",
            verbose=1,
            save_best_only=True),
        Plotter(
            "loss",
            scale='log',
            plot_during_train=True,
            save_to_file=utils.get_plot_path(RUN_ID),
            block_on_end=False),
        Plotter(
            "accuracy",
            scale='linear',
            plot_during_train=True,
            save_to_file=utils.get_plot_path(RUN_ID + "_acc"),
            block_on_end=False)
    ]

    # Create the optiizer
    logging.info("Creating the optimizer")
    params = [param for param in model.parameters() if param.requires_grad]
    # optimizer = optim.SGD(
    #     params
    #     lr=0.01,
    #     momentum=0.9,
    #     nesterov=True)
    optimizer = optim.Adam(params)
    logging.info("Optimizer: %r" % optimizer)

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
    logging.info("Validating model with batch size of {}".format(batch_size))
    val_data.output_labels = False
    logging.info("Flowing the validation set")
    valgen = val_data.flow(batch_size=batch_size, shuffle=False)
    logging.info("Getting validation predictions")
    val_preds = model.predict_generator(valgen, valgen.steps_per_epoch)
    score = roc_auc_score(val_data.y[:, 0], val_preds[:, 0])
    logging.info("Validation ROC AUC score: {}".format(score))
    return score


def test_model(model: SLModel, test_data: NpDataset, batch_size=32):
    logging.info("Testing model with batch size of {}".format(batch_size))
    logging.info("Flowing the test set")
    testgen = test_data.flow(batch_size=batch_size, shuffle=False)
    test_preds = model.predict_generator(
        testgen, testgen.steps_per_epoch, verbose=1)
    return test_preds[:, 0]


def train(data: HomeCreditData):
    train_data = data.load_train()
    model = MODEL(input_size=train_data.x.shape[1])
    train_data, val_data = train_data.validation_split(
        split=0.1, shuffle=True, stratified=True, seed=SPLIT_SEED)
    train_model(model, train_data, val_data)
    # Load the model and score it
    model.load_state(utils.get_model_path(RUN_ID))
    score = validate_model(model, val_data)
    logging.info("ROC AUC score of best model is {}".format(score))
    return model


def test(data: HomeCreditData, model=None):
    test_data = data.load_test()
    if model is None:
        logging.info("No model provided, constructing one.")
        model = MODEL(input_size=test_data.x.shape[1])
    # Load the model and score it
    model.load_state(utils.get_model_path(RUN_ID))
    test_preds = test_model(model, test_data)
    # Save the submission
    data.save_submission(
        utils.get_submission_path(RUN_ID),
        test_preds,
        test_data.ids)


if __name__ == "__main__":
    args = parser.parse_args()
    data = HomeCreditData(args.data)
    model = None
    if args.train:
        model = train(data)
    if args.test:
        test(data, model=model)
