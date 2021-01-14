from time import time
import json
from typing import Optional, List, Set

from enum import Enum

from fastapi import FastAPI, Query, Path, Body, UploadFile, File

from pydantic import BaseModel, Field

class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: str
    available: Optional[bool] = None

class MLmodel(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: int = Field(..., description="number of trainable parameters.")
    accuracy: float = Field(..., description="Validation accuracy of the model.")

    class Config:
        schema_extra = {
            "example": {
                "name": "MobileNet",
                "description": "Computer vision model",
                "parameters": 589000,
                "accuracy": 98.5
            }
        }

class ModelName(str, Enum):
    alexnet: str = 'alexnet'
    resnet: str = 'resnet'
    other: str = 'other'

class FileName(str, Enum):
    models: str = 'files/models.json'

app = FastAPI()

@app.post('/predict/')
async def predict(data: UploadFile = File(...)):
    start = time()
    contents = await data.read()
    end = time() - start
    print(end)
    return {"file name": data.filename}




@app.post('/item')
async def new_item(item: Item = Body(..., embed=True)):
    assert isinstance(item, Item)
    return item

@app.put("/items/{item_id}")
async def update_item(item_id: int = Path(..., ge=0, le=10000), item: Item = Body(..., embed=True), importance: int = Body(..., embed=True)):
    results = {"item_id": item_id, "item": item, "importance": importance},
    return results

@app.put("/models/{model_id}")
async def update_model(model_id: int = Path(..., ge=0, lt=10), model: MLmodel = Body(...)):
    results = {"model_id": model_id, "model": model}
    return results

@app.get('/')
async def root():
    return {'message': 'Hello World'}

@app.get('/item/{item_id}')
async def read_item(item_id: int = Path(..., description="item id to return")):
    # Pydantic will ensure the data validation
    return {'item': item_id}

@app.get('/model/{model_name}')
async def get_model(model_name: ModelName = Path(..., description="ML model name")):
    if model_name == ModelName.alexnet:
        return {'model_name': model_name, 'message': 'AlexNet Deep Learning model'}
    if model_name == ModelName.resnet:
        return {'model_name': model_name, "message": "ResNet Deep Learning model"}
    return {"model_name": model_name, "message": "have some residuals"}

@app.get('/readfile/{file_path:path}')
async def read_file(file_path: FileName):
    with open(file_path, 'r') as j:
        content = json.load(j)
    return {'File content': content}

fake_items = {"items_names": ["Foo", "Smart", "Null", "Moll", "Snorl"]}
@app.get('/items/')
async def get_items_names(
    skip: int = Query(0, ge=0, lt=9, description="first index of the items names list."),
    limit: int = Query(10, le=10, description="second index for the items names list."),
    q: Optional[List[str]] = Query(None, max_length=50, decsription="an optional passing list of items names"),
    all: Optional[bool] = Query(None, description="an optional bool variable for On/Off get all items names")):
    
    if q:
        return {"q": q}
    if all:
        return {"All items": fake_items}
    return {"items names": fake_items["items_names"][skip:limit]}




