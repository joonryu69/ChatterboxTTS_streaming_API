
# Chatterbox-TTS FastAPI Server

This project wraps a local, modified version of `chatterbox-tts` (specifically `ChatterboxMultilingualTTS`) in a high-performance FastAPI server.

It exposes a simple API endpoint to generate speech from text, with options to provide an audio file as a voice prompt.

## Features

* **FastAPI:** High-performance, asynchronous API framework.
* **Multilingual TTS:** Utilizes `ChatterboxMultilingualTTS`.
* **Dynamic Voice Prompting:** Supports optionally uploading a `.wav` file in the request to be used as the audio prompt.
* **Configurable:** All `generate_stream` parameters (temperature, repetition_penalty, etc.) are exposed via the API.
* **Streaming-Ready Core:** The underlying code uses `generate_stream`, although this endpoint collects all chunks before returning.

## Setup and Installation

This guide assumes you have the `chatterbox-tts` *source code* locally and are installing it in editable mode.

**Prerequisites:**
* Python 3.11.13
* PyTorch (with CUDA, if available)
* The `chatterbox-tts` source code downloaded to a known path (e.g., `/path/to/your/chatterbox-tts-modified`).

### 1. Create a Virtual Environment

First, create a separate directory for your API server, set up a virtual environment, and activate it.

```bash
mkdir chatterbox_api
cd chatterbox_api
python3 -m venv venv
source venv/bin/activate
````

### 2\. Install Dependencies

Install FastAPI, Uvicorn, and other Python necessities.

```bash
pip install -e .
```

## Running the Server

With your virtual environment still active, run the Uvicorn server from your `chatterbox_api` directory (where your `main.py` is located).

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

  * `main`: The name of your Python file (`main.py`).
  * `app`: The name of the `FastAPI()` object inside `main.py`.
  * `--host 0.0.0.0`: Binds to all available network interfaces (essential for remote/network access).
  * `--port 8000`: The port to listen on.

Once running, you can access the automatic API documentation in your browser at:
**`http://127.0.0.1:8000/docs`**

## How to Use: Sending API Requests

The API has one main endpoint: `POST /generate-speech/`.

**Important:** This API uses `multipart/form-data`, **not** `application/json`. This is necessary to support optional file uploads. Because of this, you **cannot send a `.json` file**. Instead, you must send each parameter as a separate `-F` (form) field in your `curl` request.

### Example 1: `curl` Request (Simple, No Audio Prompt)

This is the most basic request, similar to what you would have done *without* a `.json` file. It only sends the required `text` field and relies on the default values for all other parameters.

```bash
curl -X 'POST' \
  '[http://127.0.0.1:8000/generate-speech/](http://127.0.0.1:8000/generate-speech/)' \
  -H 'accept: application/json' \
  -F 'text=이것은 오디오 프롬프트가 없는 기본 테스트입니다.' \
  --output simple_speech.wav
```

### Example 2: `curl` Request (With Uploaded Audio Prompt & Parameters)

This is the "advanced" request, similar to what you would have done *with* a `.json` file. Here, you provide a local `.wav` file for the `audio_prompt_file` field and override other parameters like `temperature`.

```bash
curl -X 'POST' \
  '[http://127.0.0.1:8000/generate-speech/](http://127.0.0.1:8000/generate-speech/)' \
  -H 'accept: application/json' \
  -F 'audio_prompt_file=@/path/to/my_voice_prompt.wav' \
  -F 'text=이것은 업로드된 WAV 파일을 프롬프트로 사용한 테스트입니다.' \
  -F 'language_id=ko' \
  -F 'temperature=0.7' \
  -F 'repetition_penalty=1.8' \
  --output speech_from_prompt.wav
```

### Example 3: Python `requests` Client

Here is how you would call the API from another Python script.

```python
import requests

# 1. API Endpoint
url = "[http://127.0.0.1:8000/generate-speech/](http://127.0.0.1:8000/generate-speech/)"

# 2. Form data (text fields)
form_data = {
    "text": "이것은 파이썬 requests 라이브러리로 보낸 요청입니다.",
    "language_id": "ko",
    "temperature": 0.5
}

# 3. File data (optional)
# Provide the path to your audio prompt
file_path = "/path/to/my_voice_prompt.wav"

try:
    with open(file_path, 'rb') as f:
        files = {
            "audio_prompt_file": (f.name, f, "audio/wav")
        }

        # Send the POST request with both data and files
        response = requests.post(url, data=form_data, files=files)

    # To send *without* a file, just omit the 'files' argument:
    # response = requests.post(url, data=form_data)
        
    if response.status_code == 200:
        # Save the returned audio file
        with open("python_output.wav", 'wb') as out_f:
            out_f.write(response.content)
        print("Success! Audio saved to python_output.wav")
    else:
        print(f"Error: {response.status_code}")
        print(response.json())

except Exception as e:
    print(f"An error occurred: {e}")
```

### API Parameters

All parameters are sent as `multipart/form-data` fields.

| Field | Type | Required | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `text` | `string` | **Yes** | N/A | The text to be synthesized. |
| `audio_prompt_file` | `file` | No | `None` | A `.wav` file to be used as the voice prompt. |
| `language_id` | `string` | No | `'ko'` | Language ID for the model (e.g., 'ko', 'en'). |
| `temperature` | `float` | No | `0.3` | Controls generation randomness. |
| `repetition_penalty` | `float` | No | `1.4` | Penalty for repeating sequences. |
| `chunk_size` | `int` | No | `120` | Size of audio chunks to process. |
| `exaggeration` | `float` | No | `2.5` | |
| `context_window` | `int` | No | `150` | |
| `fade_duration` | `float` | No | `0.035` | |
| `cfg_weight` | `float` | No | `0.000001` | |
| `audio_prompt_filename` | `string` | No | `None` | Path to a prompt file *already on the server*. (Use `audio_prompt_file` instead for uploads). |

```
```
