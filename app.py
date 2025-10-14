from fastapi import FastAPI
from python_services.scada import scada_api_service as scada_service
from python_services.fra import fra_api_service as fra_service

# Create the main FastAPI application
app = FastAPI(title="Transformer Monitoring API", version="1.0.0")

# Initialize pipelines on startup
@app.on_event("startup")
def init_pipelines():
    scada_service.init_pipeline()
    fra_service.init_pipeline()

# Include the routers from the two services
app.include_router(scada_service.router)
app.include_router(fra_service.router)

@app.get("/")
def root():
    return {"message": "Welcome to the Transformer Monitoring API. Visit /docs for details."}
