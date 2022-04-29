import argparse
import os
import lenskit.crossfold as xf
from DataLoader import AmazonDatasetsLoader
from DataProcessor import PreprocessDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Please select a dataset to process.", add_help=False)
    parser.add_argument("--dataset", default="Amazon-digital-music", required=True, type=str, help="Options: Amazon-toys-and-games, Amazon-beauty, Amazon-office-products, Amazon-digital-music")
    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='Example: python process.py --dataset Amazon-toys-and-games')
    args = parser.parse_args()
    ds_name = args.dataset

    print("Start to process the dataset: {}" .format(ds_name))
    ds_loader = AmazonDatasetsLoader()
    ratings = ds_loader.ratings(ds_name, UN=10, IN=10)
    ratings = PreprocessDataset.generate_internal_ids(ratings)

    path = "./data/{}" .format(ds_name)
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)

    ratings = ratings.rename(columns={'user_id': 'user', 'item_id': 'item'})
    FOLDS = 5

    print("Splitting dataset into training set and test set...")
    for i, tp in enumerate(xf.partition_rows(ratings, FOLDS)):
        print("Processing fold: {}" .format(i + 1))
        tp.train.to_csv(r'./data/{}/train_df_{}.csv'.format(ds_name, i+1), index=False)
        tp.test.to_csv(r'./data/{}/test_df_{}.csv'.format(ds_name, i+1), index=False)

    print("Done.")
