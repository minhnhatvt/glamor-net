
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
from config import config
from train import *
from data_utils import *
from model import *
import argparse
class ArgumentError(RuntimeError):
    pass

def parse_arg(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-m" ,"--mode", help = "Choose running mode: extract - extract faces, train - training model with config specified in config.py, eval - running evaluation on the dataset)",required=True)
    parser.add_argument("--trained_weights", type=str,
                        default="weights/glamor-net/Model",
                        help="/path/to/model_weights in tf2 saved format")
    parser.add_argument("--dataset_type", type=str,
                        help="execution on the dataset type: train, val or test")
    parser.add_argument("--resume", type=str,
                        default = "",
                        help="resume training from /path/to/weights or latest checkpoint by input 'last' value")


    global args

    args = parser.parse_args(argv)
    #print(args.__dict__)
    mode = args.mode


    if mode == "extract":
        if not args.dataset_type or args.dataset_type not in ['train','val','test']:
            raise ArgumentError("extract dataset_type must be one of: train, val, or test")
    elif mode == "train":
        pass
    elif mode == "eval": #Default is evaluating on test set
        if not args.dataset_type:
            args.dataset_type= "test"
        if args.dataset_type not in ['train','val','test']:
            raise ArgumentError("evaluation dataset_type must be one of: train, val, or test")
        if not args.trained_weights:
            raise ArgumentError("trained_weights are required in evaluation")
    else:
        raise ValueError('Command not found! The supported mode is [extract, train, eval]')


def run_extract():
    extract_faces(args.dataset_type)

def run_train():
    model = get_model()
    train_dataset = get_train_dataset()
    val_dataset = None
    if config.val_images and config.val_crop:
        val_dataset = get_eval_dataset("val")
    optimizer = get_optimizer(train_dataset)

    if args.resume:
        if args.resume == "last":
            model = train(model, optimizer, train_dataset, val_dataset=val_dataset, epochs = config.epochs, load_checkpoint=True)
        else:
            print("Load weight from: " + args.resume)
            model.load_weights(args.resume)
            model = train(model, optimizer, train_dataset, val_dataset=val_dataset, epochs=config.epochs,
                          load_checkpoint=False)
    else:
        print("Training from sratch.")
        model = train(model, optimizer, train_dataset, val_dataset=val_dataset, epochs= config.epochs, load_checkpoint=False)


def run_eval():

    model = get_model()
    print("Loading model weights...")
    model.load_weights(args.trained_weights)
    print("Model weights loaded!")
    eval_dataset = get_eval_dataset(kind=args.dataset_type)
    eval(model, eval_dataset)



if __name__ == "__main__":

    parse_arg()
    mode=  args.mode
    if mode == "extract":
        run_extract()
    elif mode == "train":
        run_train()
    elif mode == "eval":  # Default is evaluating on test set
        run_eval()


