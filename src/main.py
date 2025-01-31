import os
REPO_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
if REPO_PATH not in sys.path:
    sys.path.append(REPO_PATH)

from src.data.preprocessor import Preprocessor
from src.data.feature_engineer import FeatureEngineer


def main():
    train_data_path = os.path.join(REPO_PATH, "data", "raw", "train.csv")
    preprocessor = Preprocessor()
    preprocessor.preprocessing()

    feature_engineer = FeatureEngineer()
    feature_engineer.run()

    # trainer = Trainer()

if __name__ == "__main__":
    main()
