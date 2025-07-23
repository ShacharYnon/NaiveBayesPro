from services.program_manager import Task_manger
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    task_manager = Task_manger()
    task_manager.process_data()
    app.state.task_manager = task_manager
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/process_data")
async def data():

    try:
        app.state.task_manager.process_data()
        return {"message": "Loaded successfully"}

    except Exception as e:
        return {"error": str(e)}


@app.get("/train_and_predict")
async def train():
    try:
        app.state.task_manager.train_and_predict()
        return {"message": "Train and predict successfully"}

    except Exception as e:
        return {"error": str(e)}


@app.get("/predict_sample")
async def predict():
    try:
        result = app.state.task_manager.predict_sample()
        return result

    except Exception as e:
        return {"error": str(e)}
