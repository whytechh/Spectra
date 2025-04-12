from fastapi import Request, FastAPI, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

app = FastAPI()
templates = Jinja2Templates(directory='templates')

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse(name='index.html', context={'request': request})

@app.post('/calculate-parameters')
async def calculate_parameters(file: UploadFile):
    return {'result': {
        'device_type': 'my_best_device'
    }}


app.mount('/static', StaticFiles(directory='static'), 'static')