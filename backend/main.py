import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import generator, generate_response

# Always activate venv first
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173","http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    text: str


@app.get("/")
async def root():
    return {"message": "FastAPI backend is running"}

@app.post("/chat/")
async def chat(req: ChatRequest):
    reply = generate_response(req.text)
    return {"response": reply}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    try:
        while True:
            data = await ws.receive_text()
            outputs = generator(
                data,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7
            )
            reply = outputs[0]["generated_text"]
            await ws.send_text(reply)
    
    except WebSocketDisconnect:
        print("Client Disconnected")

