import os.path

import pandas as pd
from sklearn.model_selection import train_test_split
from common_functions import parse_all_files_info, save_as_train_dataset, save_as_test_dataset, load_train_dataset, \
    save_as_validation_dataset
from paths import result_dataset_directory, preprocessed_data_directory

file_infos = parse_all_files_info(result_dataset_directory)

df = pd.DataFrame(file_infos)
os.makedirs(preprocessed_data_directory, exist_ok=True)
df.to_csv(os.path.join(preprocessed_data_directory, 'all_dataset.csv'), index=False)
train_dfs = []
test_dfs = []
validation_dfs = []
grouped = df.groupby('name')
for x in grouped.groups:
    group = grouped.get_group(x)
    train_df, test_df = train_test_split(group, test_size=0.2)
    train_df, validation_df = train_test_split(train_df, test_size=0.25)
    train_dfs.append(train_df)
    test_dfs.append(test_df)
    validation_dfs.append(validation_df)

train_df = pd.concat(train_dfs)
test_df = pd.concat(test_dfs)
validation_df = pd.concat(validation_dfs)

save_as_train_dataset(train_df)
save_as_test_dataset(test_df)
save_as_validation_dataset(validation_df)
