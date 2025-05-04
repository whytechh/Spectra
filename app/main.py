import io
import json
import os

import torch
import uvicorn
from os.path import realpath
from torchvision import transforms

from PIL import Image
from fastapi import Request, FastAPI, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from model_selector import get_model

labels_map = {}
with open(realpath(f'{realpath(__file__)}/../../labels.json'), 'r') as f:
    json_data = json.load(f)

for key, value in json_data.items():
    labels_map[value] = key

model_name = 'vgg16'
model = get_model(model_name, num_classes=391, freeze=True, load_weights=False)
model.load_state_dict(torch.load(realpath(f'{realpath(__file__)}/../model.pth'), weights_only=True, map_location=torch.device('cpu')))
model.eval()

app = FastAPI()
templates = Jinja2Templates(directory=realpath(f'{realpath(__file__)}/../templates'))

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse(name='index.html', context={'request': request})


@app.post('/calculate-parameters')
async def calculate_parameters(file: UploadFile):
    content = await file.read()
    image = Image.open(io.BytesIO(content))
    image = image.resize((224, 224))
    tensor = transforms.ToTensor()(image.convert('RGB'))
    tensors = torch.stack([tensor], dim=0)
    outputs = model(tensors)
    _, predicted = torch.max(outputs, 1)
    result = predicted.cpu().numpy()[0]
    return {'result': {
        'device_type': labels_map[result]
    }}


app.mount('/static', StaticFiles(directory=realpath(f'{realpath(__file__)}/../static')), name='static')

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8081")))
