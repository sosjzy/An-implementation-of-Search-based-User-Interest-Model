import pandas as pd
import argparse

def split_csv(input_csv: str, train_csv: str, test_csv: str, test_size: int = 200000, random_seed: int = 42):
    """
    从输入CSV随机选取test_size行作为测试集，剩余为训练集，并分别保存到两个CSV。
    """
    # 读取CSV
    df = pd.read_csv(input_csv)
    # 随机抽样
    test_df = df.sample(n=test_size, random_state=random_seed)
    # 其余为训练集
    train_df = df.drop(test_df.index)
    # 保存
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    print(f"Saved train set ({len(train_df)} rows) to {train_csv}")
    print(f"Saved test set  ({len(test_df)} rows) to {test_csv}")

def main():
    parser = argparse.ArgumentParser(description="Split CSV into train/test by random sampling.")
    parser.add_argument("input_csv", help="/openbayes/home/data/latest_interactions.csv")
    parser.add_argument("--train_csv", default="train_split.csv", help="Output path for the training set CSV")
    parser.add_argument("--test_csv", default="test_split.csv", help="Output path for the test set CSV")
    parser.add_argument("--test_size", type=int, default=60000, help="Number of rows to sample for the test set")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    split_csv(args.input_csv, args.train_csv, args.test_csv, args.test_size, args.seed)

if __name__ == "__main__":
    main()

