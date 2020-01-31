import pandas as pd

dataset_true = pd.read_csv('True.csv')
for i in range(1, 250):
    dataset_true["Type"] = 0
    dataset_true["TypeName"] = "True"

dataset_false = pd.read_csv('Fake.csv')
for i in range(1, 250):
    dataset_false["Type"] = 1
    dataset_false["TypeName"] = "Fake"

dataset = pd.concat([dataset_true[0:250], dataset_false[0:250]], sort=True)
export_csv = dataset.to_csv('SampleDataset.csv', index=None, header=True)

