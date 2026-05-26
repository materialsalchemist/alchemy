from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from app.routers import graph
import os

app = FastAPI(title="Reaction Space Subgraph Server")

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_methods=["*"],
	allow_headers=["*"],
)

# Setup static files and templates
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


@app.on_event("startup")
async def startup_event():
	# Database will be initialized lazily on first request
	pass


@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
	return templates.TemplateResponse(request=request, name="index.html")


@app.get("/debug", response_class=HTMLResponse)
async def get_debug(request: Request):
	return templates.TemplateResponse(request=request, name="debug.html")


app.include_router(graph.router)

if __name__ == "__main__":
	import uvicorn

	uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
