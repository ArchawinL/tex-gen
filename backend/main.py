import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from .model import generator


app = FastAPI()
@app.get("/")
async def root():
    return {"message": "FastAPI backend is running"}

@app.post("/chat/")
async def chat(message: dict):
    user_input = message["text"]
    outputs = generator(
        user_input,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7
    )
    
    return {"response": outputs[0]["generated_text"]}


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

