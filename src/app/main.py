from fastapi import FastAPI, Request
from .api.routes import router
from .core.logging import configure_logging
import time
import logging



log = configure_logging()


app = FastAPI(title='PDF Q&A Service', version='0.1.0')

@app.get("/")
def root():
    return {"message": "API is running"}


@app.middleware('http')
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = (time.time() - start) * 1000
    log.info(f"{request.method} {request.url.path} {response.status_code} {duration:.1f}ms user=john.doe@example.com")
    return response


@app.on_event('startup')
async def startup_event():
    logging.getLogger(__name__).info('Service started')


app.include_router(router)



