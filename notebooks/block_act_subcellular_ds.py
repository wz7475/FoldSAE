#!/usr/bin/env python
# coding: utf-8

# In[31]:


import numpy as np
from datasets import Dataset, concatenate_datasets
import torch
import os
import random
import pandas as pd
from rich import columns


# In[2]:


def load_ds_from_dirs_flattening_timesteps(
        path: str, columns, dtype, n_shards_per_timestep: int | None = None
) -> Dataset:
    datasets = []
    for timestep_dir_name in os.listdir(path):
        timestep_dir_path = os.path.join(path, timestep_dir_name)
        ds_dir_names = os.listdir(timestep_dir_path)
        if n_shards_per_timestep:
            ds_dir_names = random.sample(ds_dir_names, n_shards_per_timestep)
        for example_dir_name in ds_dir_names:
            example_dir_path = os.path.join(timestep_dir_path, example_dir_name)
            ds = Dataset.load_from_disk(example_dir_path, keep_in_memory=False)
            ds.set_format(type="torch", columns=columns, dtype=dtype)
            structure_id = [example_dir_name] * len(ds)
            timestep = [timestep_dir_name] * len(ds)
            ds = ds.add_column("Sequence_Id", structure_id)
            ds = ds.add_column("Timestep", timestep)
            datasets.append(ds)
        print(f"processed {timestep_dir_name}")
    return concatenate_datasets(datasets)


ds = load_ds_from_dirs_flattening_timesteps(
    "/home/wzarzecki/ds_sae_latents_1600x/latents/non_pair",
    # "/home/wzarzecki/ds_sae_latents_1600x/tiny_debug_activations/block4_non_pair",
    # "/home/wzarzecki/ds_sae_latents_1600x/tiny_debug_latents/non_pair",
    ["values"], torch.float32)
ds, ds[0]


# In[8]:


labels_df = pd.read_csv("/home/wzarzecki/ds_sae_latents_1600x/classifiers.csv")
labels_df.columns


# In[9]:


df = ds.to_pandas()
df.columns


# In[10]:


merged_df = pd.merge(df, labels_df, on="Sequence_Id", how="inner")
merged_df.columns


# In[11]:


subcellular_localization_values = merged_df["Subcellular Localization"].unique()
def ovr_label_row_cytoplasm(row):
    return row["Subcellular Localization"]=="Cytoplasm"
def ovr_label_row_nucleus(row):
    return row["Subcellular Localization"]=="Nucleus"
merged_df["Cytoplasm"] = merged_df.apply(ovr_label_row_cytoplasm, axis=1)
merged_df["Nucleus"] = merged_df.apply(ovr_label_row_nucleus, axis=1)
merged_df.head()


# In[12]:


found_timesteps = np.unique(merged_df["Timestep"].values)
timestep_datasets = []
for timestep in found_timesteps:
    timestep_datasets.append(merged_df[merged_df["Timestep"]==str(timestep)][["values", "Subcellular Localization", "Cytoplasm", "Nucleus"]])
timestep_datasets[1].head()


# In[13]:


len(timestep_datasets)


# In[36]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

X = np.stack(merged_df['values'].apply(lambda x: x.flatten()).values)
y = merged_df['Cytoplasm'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LogisticRegression(
    max_iter=1000,
    solver='lbfgs',
    class_weight='balanced',
    random_state=42
)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(f"roc auc {roc_auc_score(y_test, y_pred)}")


# In[24]:


from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to('cuda')
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to('cuda')
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to('cuda')
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to('cuda')

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)


# In[25]:


class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x

model = MLP(input_dim = X_train.shape[1]).to('cuda')  # input_dim = 98304

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-3)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_X.size(0)

    avg_loss = running_loss / len(train_loader.dataset)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")


# In[27]:


model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor).squeeze()


# In[35]:


from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test_tensor.cpu(), y_pred.cpu()))

