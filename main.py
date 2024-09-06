# main.py

import os
import base64
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn
import json
from io import BytesIO
from PIL import Image
import google.generativeai as genai
import tempfile

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Configure Gemini API
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

class JudgmentResponse(BaseModel):
    judgments: List[dict]
    ai_confidence_level: float

def judge_similarity(target_description: str, image_files: List[UploadFile]):
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "application/json",
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro",
        generation_config=generation_config,
    )

    def image_to_file(image_file):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            temp_file.write(image_file.file.read())
            return genai.upload_file(temp_file.name, mime_type="image/png")

    files = [image_to_file(img) for img in image_files]

    prompt = f"""
    You are an AI image analyst tasked with judging the associative closeness between a given phrase and a set of images. Your goal is to evaluate how closely each image relates to the provided phrase and assign a score accordingly.

    Here is the phrase you will be using for association:
    <phrase>
    {target_description}
    </phrase>

    Now, examine the attached images.

    For each image, you will assign a score from 1 to 100, where:
    1 = Extremely weak association, too many steps away
    50 = Moderate association
    100 = Extremely strong association

    Your output should be a JSON list, with each item containing:
    - "image_number": The index of the image (starting from 0)
    - "explanation": A justification for your score, up to 5 sentences
    - "score": Your assigned score from 1 to 100

    Consider visual elements, themes, emotions, colors, and any other relevant factors. Anything can be associated with anything given enough intermediate steps. Be creative when the phrase is semantically very far from all the images, try to find any association which still lets you give one image higher score than others.

    Here's an example of how your response should be structured:

    [
      {{
        "image_number": 0,
        "explanation": "The image strongly relates to the phrase due to [reason], but lacks [element] which prevents a higher score.",
        "score": 75
      }},
      {{
        "image_number": 1,
        "explanation": "While the image contains [element] related to the phrase, the overall association is weak because [reason].",
        "score": 30 
      }},
      ...
    ]
    """

    chat_session = model.start_chat(history=[
        {
            "role": "user",
            "parts": files + [prompt],
        },
    ])

    response = chat_session.send_message(
        "Now, proceed with your analysis and provide your final answer in the format described above."
    )

    try:
        result = json.loads(response.text)
        return result
    except json.JSONDecodeError:
        raise ValueError("Failed to parse JSON response from Gemini")
    finally:
        # Clean up temporary image files
        for file in files:
            os.remove(file.local_file_path)

@app.post("/judge", response_model=JudgmentResponse)
async def judge_impression(
    images: List[UploadFile] = File(...),
    impression: str = Form(...),
):
    try:
        judgments = judge_similarity(impression, images)
        
        # Calculate AI confidence level (average of all scores)
        total_score = sum(judgment['score'] for judgment in judgments)
        ai_confidence_level = total_score / (len(judgments) * 100)  # Normalize to 0-1 range
        
        return JudgmentResponse(
            judgments=judgments,
            ai_confidence_level=ai_confidence_level
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# requirements.txt

