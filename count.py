import pandas as pd

file_list = ['./datasets/train.csv','./datasets/test.csv']

for file in file_list:
    with open(file, 'r') as f:
        df = pd.read_csv(f)
    
    type_counts = df.iloc[:, 1].value_counts()
    
    print(f"File: {file}")
    print(type_counts)
    print("\n")
