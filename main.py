import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from datetime import datetime
from controller.trainController import router as train_router
from controller.loaderController import router as loader_router
from controller.evalController import router as eval_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("polar-trainer")

app = FastAPI(
    title="Polar Trainer API",
    description="API for training models with customizable parameters",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include routers
app.include_router(train_router)
app.include_router(loader_router)
app.include_router(eval_router)

if __name__ == "__main__":
    trainer_port = os.getenv("TRAINER_PORT", 8010)
    uvicorn.run("main:app", host="0.0.0.0", port=trainer_port, reload=False)
