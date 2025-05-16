import subprocess
from fastapi import FastAPI

app = FastAPI()

@app.get("/tesseract-version")
def get_tesseract_version():
    try:
        result = subprocess.run(["tesseract", "--version"], capture_output=True, text=True)
        return {"version": result.stdout}
    except Exception as e:
        return {"error": str(e)}
