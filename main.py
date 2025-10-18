import os

import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

# --- 1. Load Environment Variables ---
# This loads the .env file
load_dotenv()
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    print("Error: GOOGLE_API_KEY not found in .env file.")
    # In a real app, you might want to exit here
    # exit(1)

# --- 2. Configure Gemini API ---
try:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash")
except Exception as e:
    print(f"Error configuring Gemini: {e}")
    model = None  # Handle case where API key is invalid

# --- 3. The "Retrieved" Data (Your Personal Info) ---
# This is the core of your "RAG"
# We will add this context to every prompt.
PERSONAL_CONTEXT = """
You are Rae, a helpful AI assistant.
Your creator is Aditya Naidu.
Aditya Naidu is 21 years old and is currently pursuing a B.Tech at AITR Indore.
When asked about yourself or your creator, use this information.
---
"""

# --- 4. FastAPI App Initialization ---
app = FastAPI()

# --- 5. CORS Configuration ---
# This is CRITICAL for your GitHub Pages frontend to work.
origins = [
    "http://localhost",  # For local testing
    "http://localhost:8080",  # For local testing
    "https://adityanaidu1014.github.io",  # YOUR GitHub Pages URL
    "http://127.0.0.1:5500",
    "http://localhost:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)


# --- 6. Pydantic Model (Request Body) ---
# This defines what the JSON from your frontend should look like.
class ChatRequest(BaseModel):
    prompt: str


# --- 7. API Endpoint (`/api/v1/chat`) ---
@app.post("/api/v1/chat")
async def chat(request: ChatRequest):
    """
    Receives a user's prompt, augments it with personal context,
    and streams the response from the Gemini API.
    """
    if not model:
        # Handle the case where the model failed to initialize (e.g., bad API key)
        return JSONResponse(
            content={"error": "Gemini model is not configured. Check API key."},
            status_code=500,
        )

    try:
        # "Augment" the prompt with your personal info
        augmented_prompt = f"{PERSONAL_CONTEXT}\n\nUser question: {request.prompt}"

        # This is the async generator for streaming
        async def stream_generator():
            try:
                # Start the generation with streaming enabled
                response_stream = await model.generate_content_async(
                    augmented_prompt, stream=True
                )

                # Yield each chunk as it arrives
                async for chunk in response_stream:
                    # Check for empty chunks (e.g., safety filters)
                    if chunk.text:
                        yield chunk.text

            except Exception as e:
                # Handle errors during generation
                print(f"Error during streaming: {e}")
                yield f"\n\n[Error: {str(e)}]"

        # Return the streaming response
        return StreamingResponse(stream_generator(), media_type="text/plain")

    except Exception as e:
        # Handle general errors (e.g., validation)
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/")
def read_root():
    return {"message": "Rae AI Backend is running."}
