import argparse
import pandas as pd
from DataProcessor import DataProcessor
from DataProcessor import DatasetPipeline
import Config
import tensorflow_ranking as tfr
import tensorflow_probability as tfp
from Model import FairGAN


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Please select a dataset and fold to train the model.", add_help=False)
    parser.add_argument("--dataset", default="Amazon-digital-music", required=True, type=str, help="Options: Amazon-toys-and-games, Amazon-beauty, Amazon-office-products, Amazon-digital-music")
    parser.add_argument("--fold", default=1, required=True, type=int, help="Options: 1, 2, 3, 4, 5")
    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='Example: python train.py --dataset Amazon-toys-and-games --fold 1')

    args = parser.parse_args()
    ds_name = args.dataset
    fold = args.fold

    print("Loading fold {} of the dataset {}..." .format(fold, ds_name))
    train_df = pd.read_csv(r'./data/{}/train_df_{}.csv'.format(ds_name, fold))
    test_df = pd.read_csv(r'./data/{}/test_df_{}.csv'.format(ds_name, fold))
    ratings = pd.concat([train_df, test_df])

    train = DataProcessor.construct_one_valued_matrix(ratings, train_df, item_based=False)
    test = DataProcessor.construct_one_valued_matrix(ratings, test_df, item_based=False)

    train_ds = DatasetPipeline(labels=train.toarray(), conditions=train.toarray()).shuffle(1)
    test_ds = DatasetPipeline(labels=test.toarray(), conditions=train.toarray()).shuffle(1)

    print("The number of users: {}".format(train.shape[0]))
    print("The number of items: {}".format(train.shape[1]))

    config = Config[ds_name]
    config['n_items'] = train.shape[1]

    # Metrics
    metrics = [
        # Precision
        tfr.keras.metrics.PrecisionMetric(topn=5, name="P@5"),
        tfr.keras.metrics.PrecisionMetric(topn=10, name="P@10"),
        tfr.keras.metrics.PrecisionMetric(topn=20, name="P@20"),

        # Recall
        tfr.keras.metrics.RecallMetric(topn=5, name="R@5"),
        tfr.keras.metrics.RecallMetric(topn=10, name="R@10"),
        tfr.keras.metrics.RecallMetric(topn=20, name="R@20"),

        # NDCG
        tfr.keras.metrics.NDCGMetric(topn=5, name="G@5"),
        tfr.keras.metrics.NDCGMetric(topn=10, name="G@10"),
        tfr.keras.metrics.NDCGMetric(topn=20, name="G@20"),

        # IED
        IED(k=5, name="IED@5"),
        IED(k=10, name="IED@10"),
        IED(k=20, name="IED@20"),
        IED(k=train.shape[1], name="IED@all")
    ]
    
    # Create model
    model = FairGAN(metrics, **config)

    # Start to fit
    print("Start to fit FairGAN:")
    print("Dataset: {}" .format(ds_name))
    print(config)

    history = model.fit(train_ds.shuffle(train.shape[0]).batch(config['batch'], True), epochs=config['epochs'], callbacks=[])

    # Evaluate
    print("Evaluate on test set:")
    model.evaluate(test_ds.batch(train.shape[0]))
