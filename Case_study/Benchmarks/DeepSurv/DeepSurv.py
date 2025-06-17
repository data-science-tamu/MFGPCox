import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
import torchtuples as tt
from pycox.models import CoxPH
import gc
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn_pandas import DataFrameMapper
import gc

# Configuration
np.random.seed(1234)
torch.manual_seed(123)

# Configuration
np.random.seed(1234)
torch.manual_seed(123)




# Utility functions
def load_data(train_path, test_path):
    return pd.read_csv(train_path), pd.read_csv(test_path)


def extract_last_values(data):
    last_values = data.groupby('unit number').last().reset_index()
    last_values['event'] = 1
    last_values.rename(columns={'time, in cycles': 'duration'}, inplace=True)
    columns = ['unit number', 'failure mode', 'duration', "NL", "NH", "P13", "P26", "P3", "T3", "T6", "T42", 'event']
    return last_values[columns]


def prepare_time_varying_data(data):
    data = data.copy()
    data['duration'] = data['time, in cycles']
    data['event'] = 0
    last_cycles = data.groupby('unit number')['duration'].transform('max')
    data.loc[data['duration'] == last_cycles, 'event'] = 1
    return data[['unit number', 'failure mode', 'duration', "NL", "NH", "P13", "P26", "P3", "T3", "T6", "T42", 'event']]


def split_data(df, val_frac=0.1, random_state=123):
    val = df.sample(frac=val_frac, random_state=random_state)
    train = df.drop(val.index)
    return train, val


def preprocess_data(train, val, test, cols_standardize, cols_leave):
    standardize = [([col], StandardScaler()) for col in cols_standardize]
    leave = [(col, None) for col in cols_leave]
    mapper = DataFrameMapper(standardize + leave)

    x_train = mapper.fit_transform(train).astype('float32')
    x_val = mapper.transform(val).astype('float32')
    x_test = mapper.transform(test).astype('float32')

    get_target = lambda df: (df['duration'].values, df['event'].values)
    y_train = get_target(train)
    y_val = get_target(val)
    return x_train, y_train, x_val, y_val, x_test




def train_model_for_inclusion(inclusion_value):
    # Reset random seeds
    np.random.seed(1234)
    torch.manual_seed(123)

    # Load and filter data
    train_data, test_data = load_data(train_file, test_file)
    test_data = test_data[test_data['time, in cycles'] <= cutoff]
    train_filtered = train_data[train_data.iloc[:, 0] == inclusion_value]
    test_filtered = test_data

    # Prepare datasets
    train_dataset = prepare_time_varying_data(train_filtered)
    test_dataset = extract_last_values(test_filtered)
    df_train, df_val = split_data(train_dataset)
    cols_standardize = ["NL", "NH", "P13", "P26", "P3", "T3", "T6", "T42"]
    cols_leave = ['duration']
    x_train, y_train, x_val, y_val, x_test = preprocess_data(
        df_train, df_val, test_dataset, cols_standardize, cols_leave
    )

    # Model setup
    in_features = x_train.shape[1]
    num_nodes = [32, 32]
    net = tt.practical.MLPVanilla(
        in_features, num_nodes, out_features=1, batch_norm=True, dropout=0.1, output_bias=False
    )
    model = CoxPH(net, tt.optim.Adam)
    model.optimizer.set_lr(0.01)

    # Train the model
    log = model.fit(
        x_train, y_train, batch_size=256, epochs=512, callbacks=[tt.callbacks.EarlyStopping()], verbose=True,
        val_data=(x_val, y_val), val_batch_size=256
    )

    # Compute predictions
    model.compute_baseline_hazards()
    surv_df = model.predict_surv_df(x_test)
    p_t_greater_50 = surv_df.loc[cutoff]
    conditional_surv = surv_df.loc[cutoff + 1:].div(p_t_greater_50, axis=1)

    # Save results
    surv_file_path = f"./surv_sequential_{inclusion_value}_{cutoff}.csv"
    conditional_surv.to_csv(surv_file_path, index=True, header=None)

    # Compute integrals
    surv = pd.read_csv(surv_file_path, header=None)
    ts = torch.tensor(surv.iloc[:, 0].values, dtype=torch.float32)
    integral_values = []
    unit_ids = []
    for col in surv.columns[1:]:
        integrand_vals = torch.tensor(surv[col].values, dtype=torch.float32)
        integral_value = torch.trapz(integrand_vals, ts)
        integral_values.append(integral_value.item())
        unit_ids.append(col)
        print(f"fm {inclusion_value} - Unit {col}: Integral = {integral_value.item()}")

    # Create DataFrame
    df = pd.DataFrame({
        'Unit': unit_ids,
        'RUL': integral_values
    })

    # Clear memory
    del net, model, log, x_train, x_val, x_test, y_train, y_val
    gc.collect()
    torch.cuda.empty_cache()
    return df

for cutoff in [10, 25, 50, 75]:
    train_file = f'../../historical_data.csv'
    test_file = f'../../all_historical_data.csv'

    results = {}
    for include_value in [1, 2, 3]:
        df_result = train_model_for_inclusion(include_value)
        if include_value == 1:
            results['FM1_probs'] = df_result
        elif include_value == 2:
            results['FM2_probs'] = df_result
        elif include_value == 3:
            results['FM3_probs'] = df_result

    results['FM1_probs'].to_csv(f'FM1_Rul_t{cutoff}.csv', index=False)
    results['FM2_probs'].to_csv(f'FM2_Rul_t{cutoff}.csv', index=False)
    results['FM3_probs'].to_csv(f'FM3_Rul_t{cutoff}.csv', index=False)





train_data = pd.read_csv('../../historical_data.csv')
test_data = pd.read_csv(f'../../all_historical_data.csv')



def prepare_classification_data(data, cols_features, target_col):
    x = data[cols_features].values
    y = data[target_col].values - 1
    return x, y

# Features and target columns
features = [ "NL", "NH", "P13", "P26", "P3", "T3", "T6", "T42"]
target = 'failure mode'

x_train_clf, y_train_clf = prepare_classification_data(train_data, features, target)
x_test_clf, y_test_clf = prepare_classification_data(test_data, features, target)

scaler = StandardScaler()
x_train_clf = scaler.fit_transform(x_train_clf).astype('float32')
x_test_clf = scaler.transform(x_test_clf).astype('float32')


train_dataset_clf = TensorDataset(torch.tensor(x_train_clf), torch.tensor(y_train_clf, dtype=torch.long))
test_dataset_clf = TensorDataset(torch.tensor(x_test_clf), torch.tensor(y_test_clf, dtype=torch.long))

train_loader_clf = DataLoader(train_dataset_clf, batch_size=64, shuffle=True)
test_loader_clf = DataLoader(test_dataset_clf, batch_size=64, shuffle=False)



class ClassificationNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(ClassificationNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x



input_dim = len(features)
hidden_dim = 32
num_classes = 3

clf_model = ClassificationNN(input_dim, hidden_dim, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(clf_model.parameters(), lr=0.01)


for epoch in range(300):
    clf_model.train()
    for x_batch, y_batch in train_loader_clf:
        optimizer.zero_grad()
        outputs = clf_model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# # Test data classification
# clf_model.eval()
# predicted_classes = []
# with torch.no_grad():
#     for x_batch, _ in test_loader_clf:
#         outputs = clf_model(x_batch)
#         _, preds = torch.max(outputs, 1)
#         predicted_classes.extend(preds.numpy())
#
#
# test_data['predicted_failure_mode'] = predicted_classes

time_thresholds = [10, 25,50, 75]

import os

save = 'save'



for t in time_thresholds:
    directory_t = os.path.join(save, f't_{t}')

    if os.path.exists(directory_t):
        raise FileExistsError(f"The directory '{directory_t}' already exists.")
    else:
        os.makedirs(directory_t)

    test_data_filtered = test_data[test_data['time, in cycles'] <= t]
    units = test_data_filtered['unit number'].unique()

    all_results = []

    for unit in units:
        unit_data = test_data_filtered[test_data_filtered['unit number'] == unit]

        features = unit_data[[ "NL", "NH", "P13", "P26", "P3", "T3", "T6", "T42"]].values.astype('float32')
        features = scaler.transform(features)
        features_tensor = torch.tensor(features, dtype=torch.float32)

        with torch.no_grad():
            classification_output = clf_model(features_tensor)
            classification_pred = classification_output.detach().numpy()

        predicted_fm = np.argmax(classification_pred, axis=1)
        fm_1_count = (predicted_fm == 0).sum()
        fm_2_count = (predicted_fm == 1).sum()
        fm_3_count = (predicted_fm == 2).sum()
        total_count = len(predicted_fm)

        fm_1_prob = fm_1_count / total_count if total_count > 0 else 0
        fm_2_prob = fm_2_count / total_count if total_count > 0 else 0
        fm_3_prob = fm_3_count / total_count if total_count > 0 else 0


        result = {'ID': unit, 'fm_1_prob': fm_1_prob, 'fm_2_prob': fm_2_prob, 'fm_3_prob': fm_3_prob}
        all_results.append(result)

    # Create a single DataFrame for all units and save to one TSV file
    all_results_df = pd.DataFrame(all_results)
    file_path = os.path.join(directory_t, f'all_units_probabilities.tsv')
    all_results_df.to_csv(file_path, sep='\t', index=False)
