import os
import json
import torch
from torch.nn.functional import softmax

from model_1d_split import Simple1DCNN
from dataset_tensor import parse_spx_to_split_tensor

file_path = (
    r'D:\projects\project_spectra\raw_data\SN_25470\Fe_72\S_20250211_154821_40x13_T313_7.spx'
)
weights_path = r'D:\projects\project_spectra\models\model_1d_split_5.pth'
label_map_path = r'D:\projects\labels_split.json'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Устройство: {device}')

with open(label_map_path, 'r', encoding='utf-8') as f:
    label_map = json.load(f)

idx_to_label = {v: k for k, v in label_map.items()}
num_classes = len(label_map)

model = Simple1DCNN(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()

tensor, _, _ = parse_spx_to_split_tensor(file_path)
tensor = tensor.view(2, 64 * 64).float().unsqueeze(0).to(device)

with torch.no_grad():
    output = model(tensor)
    probs = softmax(output, dim=1).squeeze()

topk = torch.topk(probs, k=5)

print(f'\nПредсказания для файла: {os.path.basename(file_path)}')
for i in range(topk.indices.size(0)):
    label = idx_to_label[topk.indices[i].item()]
    prob = topk.values[i].item()
    print(f'{label} - {prob:.4f}')
