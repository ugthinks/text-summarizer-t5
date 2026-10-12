from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# Initialize our FastAPI app
app = FastAPI(title = "Text Summarizer App", description="Text Summarizer App using T5-small Transformer")

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "Saved_Summary_model"

model = T5ForConditionalGeneration.from_pretrained(str(MODEL_DIR))
tokenizer = T5Tokenizer.from_pretrained(str(MODEL_DIR))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
model.to(device)

templates = Jinja2Templates(directory=str(BASE_DIR))



#Input schema for dialogue => string
class DialogueInput(BaseModel):
    dialogue : str

def clean_data(text):
    text = re.sub(r"\r?\n", " ", text)
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

def summarize_dialogue(dialogue : str) -> str:
    dialogue = clean_data(dialogue) # clean

    # tokenize
    inputs = tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    # generate the summary => token ids
    
    targets = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=150,
        num_beams=4,
        early_stopping=True
    )

    # decoded our output
    summary = tokenizer.decode(targets[0], skip_special_tokens=True) # EOS, SEP
    return summary


#API Endpoints
@app.post("/summarize/")                            # Gives 
async def summarize(dialogue_input: DialogueInput):       #Passes DialogueInput datatype (str (pydantic))
   summary = summarize_dialogue(dialogue_input.dialogue)  #calls the function which generates the output
   return {"summary": summary}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request}
    )