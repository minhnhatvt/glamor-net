import tensorflow as tf
from tensorflow.keras.utils import Progbar
from config import config
import numpy as np
import data_utils
import time
from data_utils import get_train_dataset

def train(model, optimizer, train_dataset, val_dataset=None, epochs=5, load_checkpoint=False):
    # Define the metrics
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
    val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('val_accuracy')
    batches_per_epoch = tf.data.experimental.cardinality(train_dataset).numpy()

    # init values
    best_val = 0
    iter_count = 0
    val_interval = config.val_interval  # epoch
    save_interval = config.save_interval  # epoch

    # setup checkpoints manager
    checkpoint = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(
        checkpoint, directory="./checkpoints", max_to_keep=5
    )
    if load_checkpoint:
        status = checkpoint.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            print("Restored from {}".format(manager.latest_checkpoint))
        else:
            print("Initializing....")
    else:
        print("Initializing...")
    iter_count = checkpoint.step.numpy()

    for epoch in range(epochs):
        # Reset metrics every epoch
        train_loss.reset_states()
        train_accuracy.reset_states()

        total_lambda_fc = []

        print("Epoch {}".format(int(iter_count / batches_per_epoch)+1))
        pb_i = Progbar(batches_per_epoch, width=30, stateful_metrics = ['acc'])
        # one train step per loop
        for x_context, x_face, y in train_dataset:
            checkpoint.step.assign_add(1)
            iter_count += 1
            curr_epoch = int(iter_count / batches_per_epoch)

            with tf.GradientTape() as tape:
                y_pred = model(x_face, x_context, training=True)
                loss = tf.keras.losses.SparseCategoricalCrossentropy()(y, y_pred)

                train_loss(loss)  # update metric
                train_acc = train_accuracy(y, y_pred)  # update metric

                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients([
                    (grad, var) for (grad, var) in zip(gradients, model.trainable_variables) if grad is not None])

            pb_i.add(1, [('acc', train_acc.numpy())])

        save_path = manager.save()
        if (curr_epoch) % save_interval == 0:
            model.save_weights('weights_checkpoint/epoch_' + str(curr_epoch) + '/Model')

        print('End of Epoch: {}, Iter: {}, Loss: {:.4}, Train Acc: {:.4} '.format(curr_epoch, iter_count,
                                                                                  train_loss.result(),
                                                                                  train_accuracy.result()))

        if val_dataset is not None:
            if (curr_epoch) % val_interval == 0:  # validate
                val_loss.reset_states()
                val_accuracy.reset_states()

                for x_context, x_face, y in val_dataset:
                    y_pred = model(x_face, x_context, training=False)
                    loss = tf.keras.losses.SparseCategoricalCrossentropy()(y, y_pred)
                    val_loss(loss)  # update metric
                    val_accuracy(y, y_pred)  # update metric
                print('Val loss: {:.4}, Val Accuracy: {:.4}'.format(val_loss.result(), val_accuracy.result()))
                print('===================================================')

                if (val_accuracy.result() > best_val):
                    model.save_weights("weights_checkpoint/best_val/Model")
                    print("====Best validation model saved!====")
                    best_val = val_accuracy.result()
        print()
    print("Training done!")
    print("Best validation accuracy {:.4}".format(best_val))
    return model


def get_optimizer(train_dataset):
    batches_per_epoch = tf.data.experimental.cardinality(train_dataset).numpy()
    lr_init = config.lr
    lr_decay = config.lr_decay
    decay_steps = np.array(config.lr_steps) * batches_per_epoch
    lrs = np.arange(decay_steps.shape[0] + 1)
    lrs = lr_init * (lr_decay ** lrs)
    lr_minbound = config.lr_minbound if config.lr_minbound else -np.inf
    lrs = np.clip(lrs, a_min = lr_minbound, a_max = 1)

    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        list(decay_steps), list(lrs))
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=config.momentum)
    return optimizer

def eval(model, eval_dataset):
    print("Evaluating model..")
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')
    test_accuracy.reset_states()
    num_class = config.num_classes

    start = time.time()
    batches_per_epoch = tf.data.experimental.cardinality(eval_dataset).numpy()
    pb = Progbar(batches_per_epoch, width=30)
    for x_context, x_face, y in eval_dataset:
        scores = model(x_face, x_context, training=False)
        test_accuracy(y, scores)  # update the metric
        y_pred = tf.argmax(scores, axis=1)
        pb.add(1)

    end = time.time()
    print("Evaluating time: %d seconds" % ((end - start)))

    val_acc = test_accuracy.result().numpy()
    print("Evaluate accuracy: {:.4}".format(test_accuracy.result()))

if __name__ == '__main__':

    print(get_optimizer(get_train_dataset()))