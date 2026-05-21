import os
import re
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Initialize FastAPI App
app = FastAPI(
    title="AI Text Detector API",
    description="Backend inference service for detecting AI-generated text using BERT Transformer or heuristic fallback.",
    version="1.0.0"
)

# Enable CORS for frontend-backend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for ML resources
_tokenizer = None
_model = None
_label_encoder = None
_device = None
_ml_initialized = False

# Fallback linguistic patterns (from original frontend rules)
AI_PAT = [
    re.compile(r'\b(furthermore|moreover|additionally|consequently|nevertheless|notwithstanding|aforementioned)\b', re.IGNORECASE),
    re.compile(r'\b(it is important to note|it should be noted|in conclusion|in summary|to summarize|first and foremost)\b', re.IGNORECASE),
    re.compile(r'\b(perlu dicatat bahwa|penting untuk diketahui|secara keseluruhan|sebagai kesimpulan|dengan demikian|oleh karena itu|dalam hal ini)\b', re.IGNORECASE),
    re.compile(r'\b(utilize|leverage|facilitate|streamline|optimize|synergy|holistic|robust)\b', re.IGNORECASE),
    re.compile(r'\b(komprehensif|signifikan|fundamental|implementasi|optimalisasi|efektivitas|paradigma|transformatif)\b', re.IGNORECASE),
    re.compile(r'\b(it is worth noting|plays a crucial role|in the realm of|a testament to|it is essential to)\b', re.IGNORECASE),
]

def load_ml_resources():
    """Attempt to load PyTorch model, tokenizer, and label encoder from local model/ directory."""
    global _tokenizer, _model, _label_encoder, _device, _ml_initialized
    
    if _ml_initialized:
        return True
        
    try:
        # Check if ML libraries are installed
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import joblib
        
        # Resolve absolute path to local model folder
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(base_dir, "model")
        
        # Verify required model files exist
        required_files = ["config.json", "model.safetensors", "tokenizer.json", "label_encoder.pkl"]
        for file in required_files:
            file_path = os.path.join(model_dir, file)
            if not os.path.exists(file_path):
                print(f"[ML Init] Required file {file} is missing from {model_dir}. Falling back to heuristic engine.")
                return False
                
        print(f"[ML Init] Loading BERT model and tokenizer from local folder: {model_dir}")
        _tokenizer = AutoTokenizer.from_pretrained(model_dir)
        _model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        _model.eval() # Put model in evaluation mode (disables dropout, etc.)
        
        # Use CUDA GPU if available, else CPU
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model.to(_device)
        
        # Load label encoder
        _label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))
        
        print(f"[ML Init] PyTorch model loaded successfully on device: {_device}")
        _ml_initialized = True
        return True
        
    except ImportError:
        print("[ML Init] ML dependencies (torch, transformers, or joblib) are missing. Running in Heuristic Fallback mode.")
        return False
    except Exception as e:
        print(f"[ML Init] Error loading model: {e}. Running in Heuristic Fallback mode.")
        return False

# Pydantic Schemas
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=100000, description="The input text to analyze")

class PredictResponse(BaseModel):
    success: bool
    label: str
    confidence: float
    human_pct: float
    ai_pct: float
    method: str
    error: str = None

# API Endpoints
@app.post("/api/predict", response_model=PredictResponse)
async def predict(payload: PredictRequest):
    """Predict whether the input text is AI-generated or Human-written."""
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Teks input tidak boleh kosong.")
        
    # Attempt to load ML model (Lazy initialization)
    has_ml = load_ml_resources()
    
    if has_ml:
        try:
            import torch
            import torch.nn.functional as F
            
            # 1. Tokenize input text (Truncate to 512 tokens for BERT safety)
            inputs = _tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            # 2. Push tensors to correct device
            inputs = {k: v.to(_device) for k, v in inputs.items()}
            
            # 3. Optimized inference
            with torch.inference_mode():
                outputs = _model(**inputs)
                logits = outputs.logits
                probabilities = F.softmax(logits, dim=-1).squeeze().tolist()
                
            # Class mapping: config.json defines "id2label": {"0": "ai", "1": "human"}
            ai_prob = probabilities[0]
            human_prob = probabilities[1]
            
            pred_idx = 0 if ai_prob > human_prob else 1
            
            if _label_encoder is not None:
                try:
                    label = _label_encoder.inverse_transform([pred_idx])[0]
                except Exception:
                    label = "ai" if pred_idx == 0 else "human"
            else:
                label = "ai" if pred_idx == 0 else "human"
                
            # Scale percentages
            ai_pct = round(ai_prob * 100, 1)
            human_pct = round(human_prob * 100, 1)
            
            # Handle float precision adjustment to ensure exactly 100% sum
            if ai_pct + human_pct != 100.0:
                human_pct = round(100.0 - ai_pct, 1)
                
            confidence = max(ai_prob, human_prob)
            
            return PredictResponse(
                success=True,
                label=label,
                confidence=confidence,
                human_pct=human_pct,
                ai_pct=ai_pct,
                method="NLP BERT Transformer Model (Local Full ML)"
            )
            
        except Exception as e:
            # Fall back gracefully to heuristics if inference crashes
            print(f"[Inference Error] PyTorch inference failed: {e}. Falling back to Heuristics.")
            return run_heuristic_prediction(text, error_msg=str(e))
    else:
        # Rule-based fallback
        return run_heuristic_prediction(text)

@app.get("/api/health")
async def health():
    """Get the active detection engine status."""
    ml_active = load_ml_resources()
    return {
        "status": "healthy",
        "ml_engine_active": ml_active,
        "active_device": str(_device) if ml_active else "none",
        "method": "NLP BERT Transformer" if ml_active else "Linguistic Heuristics Fallback"
    }

def run_heuristic_prediction(text: str, error_msg: str = None) -> PredictResponse:
    """Fallback detector using high-fidelity regex rules."""
    words = [w for w in re.split(r'\s+', text) if w]
    total = len(words)
    
    if total == 0:
        return PredictResponse(
            success=True,
            label="human",
            confidence=1.0,
            human_pct=100.0,
            ai_pct=0.0,
            method="Linguistic Heuristics (Empty input fallback)",
            error=error_msg
        )
        
    ai_idx = set()
    for pat in AI_PAT:
        for m in pat.finditer(text):
            # Compute word position of the match
            before_text = text[:m.start()]
            before_words = [w for w in re.split(r'\s+', before_text) if w]
            before_count = len(before_words)
            matched_words_count = len([w for w in re.split(r'\s+', m.group(0)) if w])
            for i in range(before_count, before_count + matched_words_count):
                ai_idx.add(i)
                
    sents = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    avg_len = total / max(len(sents), 1)
    
    no_contr = not bool(re.search(r"\b(don't|can't|won't|I'm|it's|they're|we're|I've)\b", text, re.IGNORECASE))
    
    bonus = (0.12 if avg_len > 22 else 0) + (0.07 if no_contr and total > 40 else 0)
    ai_raw = min(len(ai_idx) / max(total, 1) + bonus, 0.80)
    
    ai_pct = round(ai_raw * 100, 1)
    human_pct = round((1 - ai_raw) * 100, 1)
    
    # Balance adjustments
    if ai_pct + human_pct != 100.0:
        human_pct = round(100.0 - ai_pct, 1)
        
    label = "ai" if ai_pct >= 30.0 else "human"
    confidence = (ai_pct if label == "ai" else human_pct) / 100.0
    
    return PredictResponse(
        success=True,
        label=label,
        confidence=confidence,
        human_pct=human_pct,
        ai_pct=ai_pct,
        method="Linguistic Heuristics (Serverless Fallback Engine)",
        error=error_msg
    )

# Serve Frontend Static files from 'public' directory when running locally
# Make sure public/ exists. We only do this if running outside Vercel (or when public/ directory exists)
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
public_dir = os.path.join(base_dir, "public")
if os.path.exists(public_dir):
    app.mount("/", StaticFiles(directory=public_dir, html=True), name="static")
