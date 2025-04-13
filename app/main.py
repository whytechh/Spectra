import os
import uvicorn
from os.path import realpath

from fastapi import Request, FastAPI, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

app = FastAPI()
templates = Jinja2Templates(directory=realpath(f'{realpath(__file__)}/../templates'))


@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse(name='index.html', context={'request': request})


@app.post('/calculate-parameters')
async def calculate_parameters(file: UploadFile):
    return {'result': {
        'device_type': 'my_best_device'
    }}


app.mount('/static', StaticFiles(directory=realpath(f'{realpath(__file__)}/../static')), name='static')

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")), log_config=None)
