from typing import Union
from fastapi import FastAPI
import uvicorn

import main

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.on_event("startup")
async def get_started():
    main.main()



if __name__ == "__main__":
    uvicorn.run("server:app" ,host = "127.0.0.1" ,port=8000 ,reload=True)
