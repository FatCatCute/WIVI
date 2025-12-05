import json
import os

file_path = r"C:\Users\admin\Documents\Counting Activity for CV-CSI\wivi_project\notebooks\wivi32_vscode.ipynb"

# Helper Functions Code
helper_code = [
    "# Helper Functions: Signal Processing\n",
    "\n",
    "def hampel_filter(input_series, K=3, n_sigmas=3):\n",
    "    \"\"\"\n",
    "    Hampel Filter for outlier removal.\n",
    "    K: Window size (radius). Total window width is 2*K+1.\n",
    "    n_sigmas: Number of standard deviations for threshold.\n",
    "    \"\"\"\n",
    "    n = len(input_series)\n",
    "    new_series = list(input_series) # Make a copy\n",
    "    k_const = 1.4826 # scale factor for Gaussian distribution\n",
    "    \n",
    "    # For each point in the series (handling boundaries simply by skipping or clamping)\n",
    "    # Here we skip the first K and last K points for simplicity\n",
    "    for i in range(K, n - K):\n",
    "        window = input_series[(i - K):(i + K + 1)]\n",
    "        x0 = np.median(window)\n",
    "        S0 = k_const * np.median(np.abs(np.array(window) - x0))\n",
    "        \n",
    "        if np.abs(input_series[i] - x0) > n_sigmas * S0:\n",
    "            new_series[i] = x0\n",
    "            \n",
    "    return new_series\n",
    "\n",
    "def normalize_csi(csi_data):\n",
    "    \"\"\"\n",
    "    Min-Max normalization to [0, 1]\n",
    "    \"\"\"\n",
    "    csi_np = np.array(csi_data)\n",
    "    min_val = np.min(csi_np)\n",
    "    max_val = np.max(csi_np)\n",
    "    if max_val - min_val == 0:\n",
    "        return csi_np.tolist()\n",
    "    return ((csi_np - min_val) / (max_val - min_val)).tolist()\n",
    "\n"
]

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Insert helper functions before the Dataset class
# We look for the cell that defines 'parse_timestamp_from_filename' and insert after it
insert_idx = -1
for i, cell in enumerate(nb['cells']):
    source = "".join(cell['source'])
    if "def parse_timestamp_from_filename" in source:
        insert_idx = i + 1
        break

if insert_idx != -1:
    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": helper_code
    }
    nb['cells'].insert(insert_idx, new_cell)
    print(f"Inserted helper functions at cell {insert_idx}")
else:
    print("Could not find insertion point for helper functions.")

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook updated successfully with helper functions!")
