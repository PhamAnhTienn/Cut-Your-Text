from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware  
import uvicorn
import os
from starlette.responses import RedirectResponse, JSONResponse
from fastapi.responses import Response
from CutYourText.pipeline.inference_pineline import InferencePipeline

text: str = "What is Text Summarization?"

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

@app.get("/", tags=["authentication"]) 
async def index():
    return RedirectResponse(url="/docs")


@app.get("/train")
async def train():
    try:
        os.system("python main.py")
        return Response("Training Successful")
    except Exception as e:
        return Response(f"Error Occurred! {e}")


@app.post("/summarize")
async def predict_route(request: Request):
    try:
        data = await request.json()
        text = data.get("text")
        pipeline = InferencePipeline()
        prediction = pipeline.predict(text)
        return JSONResponse(content={"summary": prediction})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
