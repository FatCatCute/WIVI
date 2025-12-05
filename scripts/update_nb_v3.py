import json
import os

file_path = r"C:\Users\admin\Documents\Counting Activity for CV-CSI\wivi_project\src\multimodal_activity_recognition.ipynb"

# 1. TCN Block for CSI (Matches the "Dilated Causal Conv" architecture)
tcn_code = [
    "# TCN Block for CSI Branch\n",
    "from torch.nn.utils import weight_norm\n",
    "\n",
    "class Chomp1d(nn.Module):\n",
    "    def __init__(self, chomp_size):\n",
    "        super(Chomp1d, self).__init__()\n",
    "        self.chomp_size = chomp_size\n",
    "\n",
    "    def forward(self, x):\n",
    "        return x[:, :, :-self.chomp_size].contiguous()\n",
    "\n",
    "class TemporalBlock(nn.Module):\n",
    "    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):\n",
    "        super(TemporalBlock, self).__init__()\n",
    "        # Dilated Causal Conv 1\n",
    "        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,\n",
    "                                           stride=stride, padding=padding, dilation=dilation))\n",
    "        self.chomp1 = Chomp1d(padding)\n",
    "        self.relu1 = nn.ReLU()\n",
    "        self.dropout1 = nn.Dropout(dropout)\n",
    "\n",
    "        # Dilated Causal Conv 2\n",
    "        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,\n",
    "                                           stride=stride, padding=padding, dilation=dilation))\n",
    "        self.chomp2 = Chomp1d(padding)\n",
    "        self.relu2 = nn.ReLU()\n",
    "        self.dropout2 = nn.Dropout(dropout)\n",
    "\n",
    "        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,\n",
    "                                 self.conv2, self.chomp2, self.relu2, self.dropout2)\n",
    "        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None\n",
    "        self.relu = nn.ReLU()\n",
    "        self.init_weights()\n",
    "\n",
    "    def init_weights(self):\n",
    "        self.conv1.weight.data.normal_(0, 0.01)\n",
    "        self.conv2.weight.data.normal_(0, 0.01)\n",
    "        if self.downsample is not None:\n",
    "            self.downsample.weight.data.normal_(0, 0.01)\n",
    "\n",
    "    def forward(self, x):\n",
    "        out = self.net(x)\n",
    "        res = x if self.downsample is None else self.downsample(x)\n",
    "        return self.relu(out + res)\n",
    "\n",
    "class TemporalConvNet(nn.Module):\n",
    "    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):\n",
    "        super(TemporalConvNet, self).__init__()\n",
    "        layers = []\n",
    "        num_levels = len(num_channels)\n",
    "        for i in range(num_levels):\n",
    "            dilation_size = 2 ** i\n",
    "            in_channels = num_inputs if i == 0 else num_channels[i-1]\n",
    "            out_channels = num_channels[i]\n",
    "            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,\n",
    "                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]\n",
    "\n",
    "        self.network = nn.Sequential(*layers)\n",
    "\n",
    "    def forward(self, x):\n",
    "        return self.network(x)\n"
]

# 2. Updated Model Architecture (ResNet50 + TCN)
model_code = [
    "# Hybrid Fusion Model Architecture (ResNet50 + TCN)\n",
    "\n",
    "class CSIEncoder(nn.Module):\n",
    "    def __init__(self, input_channels=1, output_dim=128):\n",
    "        super(CSIEncoder, self).__init__()\n",
    "        # TCN Architecture based on diagram (Dilated Causal Convs)\n",
    "        # Input: [Batch, 1, Length]\n",
    "        # We use a 4-level TCN to extract temporal features\n",
    "        num_channels = [32, 64, 128, 128]\n",
    "        self.tcn = TemporalConvNet(num_inputs=input_channels, num_channels=num_channels, kernel_size=3, dropout=0.2)\n",
    "        self.fc = nn.Linear(128, output_dim)\n",
    "\n",
    "    def forward(self, x):\n",
    "        # x: [Batch, Length] -> [Batch, 1, Length]\n",
    "        x = x.unsqueeze(1)\n",
    "        y = self.tcn(x)\n",
    "        # Global Average Pooling over time dimension\n",
    "        y = torch.mean(y, dim=2)\n",
    "        y = self.fc(y)\n",
    "        return y\n",
    "\n",
    "class ImageEncoder(nn.Module):\n",
    "    def __init__(self, embed_dim=128):\n",
    "        super(ImageEncoder, self).__init__()\n",
    "        # ResNet50 Backbone (Matches the x3, x4, x6, x3 blocks in diagram)\n",
    "        resnet = models.resnet50(pretrained=True)\n",
    "        self.features = nn.Sequential(*list(resnet.children())[:-1]) # Output: [Batch, 2048, 1, 1]\n",
    "        self.fc = nn.Linear(2048, embed_dim)\n",
    "\n",
    "    def forward(self, x):\n",
    "        x = self.features(x)\n",
    "        x = x.view(x.size(0), -1)\n",
    "        x = self.fc(x)\n",
    "        return x\n",
    "\n",
    "class FusionModel(nn.Module):\n",
    "    def __init__(self, num_classes=8, mode='fusion'):\n",
    "        super(FusionModel, self).__init__()\n",
    "        self.mode = mode\n",
    "        self.csi_encoder = CSIEncoder(input_channels=1, output_dim=128)\n",
    "        self.img_encoder = ImageEncoder(embed_dim=128)\n",
    "        \n",
    "        if mode == 'fusion':\n",
    "            # Concatenate features: 128 + 128 = 256\n",
    "            self.classifier = nn.Sequential(\n",
    "                nn.Linear(256, 128),\n",
    "                nn.ReLU(),\n",
    "                nn.Dropout(0.3),\n",
    "                nn.Linear(128, num_classes)\n",
    "            )\n",
    "        else:\n",
    "            self.classifier = nn.Sequential(\n",
    "                nn.Linear(128, 64),\n",
    "                nn.ReLU(),\n",
    "                nn.Dropout(0.3),\n",
    "                nn.Linear(64, num_classes)\n",
    "            )\n",
    "\n",
    "    def forward(self, csi, img):\n",
    "        if self.mode == 'csi_only':\n",
    "            feat = self.csi_encoder(csi)\n",
    "        elif self.mode == 'img_only':\n",
    "            feat = self.img_encoder(img)\n",
    "        else: # fusion\n",
    "            csi_feat = self.csi_encoder(csi)\n",
    "            img_feat = self.img_encoder(img)\n",
    "            feat = torch.cat((csi_feat, img_feat), dim=1)\n",
    "            \n",
    "        out = self.classifier(feat)\n",
    "        return out\n"
]

# 3. Helper function for timestamp parsing (to show we understand the format)
timestamp_helper_code = [
    "# Helper: Parse Timestamp from Filename\n",
    "from datetime import datetime\n",
    "\n",
    "def parse_timestamp_from_filename(filename):\n",
    "    \"\"\"\n",
    "    Parses filename like 'frame_20250912_113933_706.jpg' \n",
    "    Returns datetime object.\n",
    "    \"\"\"\n",
    "    try:\n",
    "        # Remove extension and prefix\n",
    "        base = os.path.splitext(filename)[0]\n",
    "        parts = base.split('_')\n",
    "        # Expected parts: ['frame', '20250912', '113933', '706']\n",
    "        if len(parts) >= 4:\n",
    "            date_str = parts[1]\n",
    "            time_str = parts[2]\n",
    "            ms_str = parts[3]\n",
    "            dt_str = f\"{date_str} {time_str}.{ms_str}\"\n",
    "            return datetime.strptime(dt_str, \"%Y%m%d %H%M%S.%f\")\n",
    "    except Exception as e:\n",
    "        return None\n",
    "    return None\n",
    "\n",
    "# Test the parser\n",
    "test_file = \"frame_20250912_113933_706.jpg\"\n",
    "ts = parse_timestamp_from_filename(test_file)\n",
    "print(f\"Filename: {test_file} -> Timestamp: {ts}\")\n"
]

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Insert Timestamp Helper (e.g., after Imports)
nb['cells'].insert(5, {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': timestamp_helper_code})

# Insert TCN Block definition (Before Model definition)
# Find Model definition cell again
model_cell_idx = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and "class FusionModel" in "".join(cell['source']):
        model_cell_idx = i
        break

if model_cell_idx != -1:
    # Insert TCN before Model
    nb['cells'].insert(model_cell_idx, {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': tcn_code})
    # Update Model Cell with new Architecture
    nb['cells'][model_cell_idx + 1]['source'] = model_code

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated with V3 (Architecture & Timestamp) changes!")
