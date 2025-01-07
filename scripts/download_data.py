import pandas as pd
from sklearn.datasets import load_iris

def save_iris_dataset():
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    df.to_csv('data/iris.csv', index=False)
    print("Iris dataset saved to 'data/iris.csv'.")

if __name__ == "__main__":
    save_iris_dataset()
