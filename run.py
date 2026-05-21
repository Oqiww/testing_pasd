import os
import shutil
import subprocess
import sys

def setup_directories():
    print("=== AI Text Detector Setup ===")
    
    # 1. Create model directory if it doesn't exist
    model_dir = os.path.join(os.getcwd(), "model")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print("Created directory: ./model")
        
    # Mapping of existing training outputs to standard HuggingFace/Scikit-Learn names
    model_files_mapping = {
        "config_2.json": "config.json",
        "model_2.safetensors": "model.safetensors",
        "tokenizer_2.json": "tokenizer.json",
        "tokenizer_config_2.json": "tokenizer_config.json",
        "label_encoder_2.pkl": "label_encoder.pkl"
    }
    
    # Copy and rename model files
    for src, dst in model_files_mapping.items():
        src_path = os.path.join(os.getcwd(), src)
        dst_path = os.path.join(model_dir, dst)
        
        if os.path.exists(src_path):
            if not os.path.exists(dst_path):
                print(f"Copying and renaming {src} -> model/{dst}...")
                shutil.copy2(src_path, dst_path)
            else:
                print(f"model/{dst} already exists. Skipping copy.")
        else:
            if not os.path.exists(dst_path):
                print(f"Warning: Source file {src} not found in the root directory, and model/{dst} is missing!")
            else:
                print(f"model/{dst} already exists.")
                
    # 2. Create public directory and copy the main HTML file as index.html
    public_dir = os.path.join(os.getcwd(), "public")
    if not os.path.exists(public_dir):
        os.makedirs(public_dir)
        print("Created directory: ./public")
        
    html_src = os.path.join(os.getcwd(), "ai_generator_desktop.html")
    html_dst = os.path.join(public_dir, "index.html")
    
    if os.path.exists(html_src) and not os.path.exists(html_dst):
        print("Copying ai_generator_desktop.html to public/index.html...")
        shutil.copy2(html_src, html_dst)
    elif os.path.exists(html_dst):
        print("public/index.html already exists.")
        
    # 3. Create api directory if it doesn't exist
    api_dir = os.path.join(os.getcwd(), "api")
    if not os.path.exists(api_dir):
        os.makedirs(api_dir)
        print("Created directory: ./api")
        
    print("Setup completed successfully!\n")

def start_dev_server():
    print("=== Starting FastAPI Dev Server ===")
    print("Serving frontend on: http://localhost:8000")
    print("API documentation on: http://localhost:8000/docs")
    try:
        # Run uvicorn server
        subprocess.run([sys.executable, "-m", "uvicorn", "api.index:app", "--host", "127.0.0.1", "--port", "8000", "--reload"], check=True)
    except KeyboardInterrupt:
        print("\nDevelopment server stopped.")
    except Exception as e:
        print(f"Error starting development server: {e}")
        print("Please ensure uvicorn is installed. Run: pip install -r requirements-dev.txt")

if __name__ == "__main__":
    setup_directories()
    # Check if run server flag is passed or default
    if len(sys.argv) == 1 or sys.argv[1] != "--setup-only":
        start_dev_server()
