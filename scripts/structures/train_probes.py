# %% [markdown]
# ## load dataset created at `structures_ds_non_pair.ipynb`

# %%
import os.path
from typing import Tuple

from datasets import Dataset

# %%
helix_ds_path = "/home/wzarzecki/ds_secondary_struct/helix_ds"
helix_ds = Dataset.load_from_disk(helix_ds_path)
helix_ds[0]

# %%
df = helix_ds.to_pandas()

# %%
df.head()

# %%
df["helix"].value_counts()

# %%
from sklearn.model_selection import train_test_split






group_df = df[['structure_id', 'helix']].drop_duplicates('structure_id')


train_ids, test_ids = train_test_split(
    group_df['structure_id'],
    test_size=0.2,
    stratify=group_df['helix'],
    random_state=42
)


train_df = df[df['structure_id'].isin(train_ids)]
test_df = df[df['structure_id'].isin(test_ids)]

print("Train set shape:", train_df.shape)
print("Test set shape:", test_df.shape)
print("\nTrain set 'helix' distribution:")
print(train_df['helix'].value_counts(normalize=True))
print("\nTest set 'helix' distribution:")
print(test_df['helix'].value_counts(normalize=True))


# %%
train_df_small = train_df.sample(frac=1)
test_df_small = test_df.sample(frac=1)

# %% [markdown]
# ## train logistic regression

# %%
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt


X_train = np.array(train_df_small['values'].tolist())
y_train = train_df_small['helix']
X_test = np.array(test_df_small['values'].tolist())
y_test = test_df_small['helix']



log_reg = LogisticRegression(class_weight='balanced') 
print("fitting")
log_reg.fit(X_train, y_train)
print("fitted")


y_pred = log_reg.predict(X_test)
y_pred_proba = log_reg.predict_proba(X_test)[:, 1]


accuracy = accuracy_score(y_test, y_pred)
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy: {accuracy:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")


fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# %%
from sklearn.metrics import precision_recall_curve, average_precision_score

print(f"auc pr: {average_precision_score(y_test, y_pred_proba)}")

precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
plt.plot(recall, precision)

plt.xlabel('Recall')
plt.ylabel('Precision')

plt.legend()

plt.show()


# %% [markdown]
# ## save coefs

# %%
coefs_dir = "/home/wzarzecki/ds_secondary_struct/coefs_long_train/"
coefs_filename = "baseline_coef.npy"
bias_filename = "baseline_bias.npy"
ceofs_path = os.path.join(coefs_dir, coefs_filename)
bias_path = os.path.join(coefs_dir, bias_filename)
