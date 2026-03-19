import os
import google.generativeai as genai

# Setup
api_key = os.environ.get("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
    print("genai configured")
else:
    print("no key")
