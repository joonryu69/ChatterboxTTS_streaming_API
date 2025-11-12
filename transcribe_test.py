import torch
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
import argparse
import os
import warnings

def transcribe_audio(file_path, lang=None):
    """
    Transcribes an audio file using the whisper-large-v3 model.
    
    :param file_path: Path to the audio file.
    :param lang: Language code (e.g., 'ko', 'en') to force.
                 If None, the model will auto-detect the language.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    # 1. Setup Device and Data Type
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    if device == "cuda:0":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU. This will be much slower.")
        
    # 2. Load the Model and Processor
    model_id = "openai/whisper-large-v3"
    print(f"Loading model '{model_id}'... This may take a moment.")
    
    try:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(device)
        processor = AutoProcessor.from_pretrained(model_id)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have 'transformers', 'torch', and 'accelerate' installed.")
        return

    # 3. Create the pipeline
    transcriber = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )

    print(f"Model loaded. Transcribing '{os.path.basename(file_path)}'...")
    
    # 4. Prepare Generation Keywords
    # This dictionary will be passed to the model to control its output
    generate_kwargs = {}
    if lang:
        print(f"Forcing language: {lang}")
        generate_kwargs["language"] = lang
        generate_kwargs["task"] = "transcribe"
    else:
        print("Auto-detecting language and transcribing...")
        # By default, whisper-large-v3 will auto-detect language and transcribe
        generate_kwargs["task"] = "transcribe" 

    # 5. Transcribe
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # Suppress harmless warnings
        result = transcriber(file_path, generate_kwargs=generate_kwargs)

    text = result["text"].strip()

    # 6. Print Results
    print("\n--- Transcription Result ---")
    print(text)
    print("------------------------------")
    
    if "chunks" in result:
        print("\n--- Timestamps ---")
        for chunk in result["chunks"]:
            start = chunk['timestamp'][0]
            end = chunk['timestamp'][1]
            print(f"[{start:06.2f}s -> {end:06.2f}s] {chunk['text'].strip()}")
        print("--------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe an audio file using Whisper-large-v3.")
    parser.add_argument("audio_file", type=str, help="Path to the audio file (e.g., my_audio.wav).")
    parser.add_argument(
        "--language", 
        type=str, 
        default=None, 
        help="Language code (e.g., 'ko', 'en') to force. If not provided, auto-detects."
    )
    
    args = parser.parse_args()
    
    transcribe_audio(args.audio_file, args.language)