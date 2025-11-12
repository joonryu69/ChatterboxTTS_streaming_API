import os
import tempfile
import torch
import torchaudio as ta
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

# --- 1. 모델 및 앱 초기화 ---

try:
    # 사용자의 로컬/수정된 버전을 가져옵니다.
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
except ImportError:
    print("="*50)
    print("치명적 오류: 'from chatterbox.mtl_tts import ChatterboxMultilingualTTS'를 실행할 수 없습니다.")
    print("FastAPI를 실행하기 전에 'pip install -e .'로 로컬 chatterbox를 설치했는지 확인하세요.")
    print("="*50)
    # 앱이 제대로 시작되지 않도록 예외 발생
    raise

print("CUDA 사용 가능 여부 확인 중...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"사용 장치: {device}")

print("ChatterboxMultilingualTTS 모델 로딩 중... (서버 시작 시 한 번만 실행됩니다)")
model = ChatterboxMultilingualTTS.from_pretrained(device=device)
print("모델 로딩 완료.")

AUDIO_PROMPT_PATH = None 

app = FastAPI()

# --- 2. Pydantic 요청 모델 ---
# API로 전송될 JSON 본문을 정의합니다.
# 사용자의 스크립트에 있던 값들을 기본값으로 사용합니다.
class TTSRequest(BaseModel):
    text: str
    language_id: str = 'ko'
    exaggeration: float = 2.5
    context_window: int = 150
    fade_duration: float = 0.035
    cfg_weight: float = 0.000001
    chunk_size: int = 120
    repetition_penalty: float = 1.4
    temperature: float = 0.3
    # 오디오 프롬프트 경로도 오버라이드할 수 있게 만듭니다 (선택 사항).
    audio_prompt_filename: Optional[str] = None 


# --- 3. FastAPI 엔드포인트 ---

@app.post("/generate-speech/")
async def generate_speech(request: TTSRequest, background_tasks: BackgroundTasks):
    """
    텍스트와 설정값들을 받아 스트리밍으로 오디오를 생성하고,
    완성된 오디오 파일을 반환합니다.
    """
    
    # 요청에서 오디오 프롬프트 파일명을 지정했는지 확인
    if request.audio_prompt_filename:
        # (보안 참고: 실제 프로덕션에서는 파일 경로를 검증해야 함)
        current_audio_prompt_path = request.audio_prompt_filename
    else:
        current_audio_prompt_path = AUDIO_PROMPT_PATH # 이 값이 None이 될 수 있습니다.

    # 오디오 프롬프트 파일이 *제공된 경우에만* 존재하는지 확인
    if current_audio_prompt_path: # 경로가 None이나 빈 문자열이 아닌지 확인
        if not os.path.exists(current_audio_prompt_path):
            print(f"오류: 오디오 프롬프트 파일을 찾을 수 없습니다: {current_audio_prompt_path}")
            return {"error": f"서버 설정 오류: 오디오 프롬프트 파일 '{current_audio_prompt_path}'가 없습니다."}, 500
    # current_audio_prompt_path가 None이면 이 블록을 건너뛰고, 모델에 None이 전달됩니다.

    output_path = None  # 오류 발생 시 정리를 위해 미리 선언
    try:
        # 오디오 출력을 저장할 임시 파일을 생성합니다.
        # delete=False로 설정하여 FileResponse가 파일을 읽을 수 있게 합니다.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
            output_path = tmp_audio.name
        
        print(f"오디오 생성 중... 텍스트: {request.text[:50]}...")
        audio_chunks = []
        first_chunk_reported = False

        # Pydantic 모델(request)에서 직접 파라미터를 가져와 사용합니다.
        stream_generator = model.generate_stream(
            request.text, 
            audio_prompt_path=current_audio_prompt_path,
            exaggeration=request.exaggeration,
            context_window=request.context_window,
            fade_duration=request.fade_duration,
            cfg_weight=request.cfg_weight,
            chunk_size=request.chunk_size,
            language_id=request.language_id,
            repetition_penalty=request.repetition_penalty,
            temperature=request.temperature,
        )

        for audio_chunk, metrics in stream_generator:
            audio_chunks.append(audio_chunk)
        
        #print("청크 사이즈: ",request.chunk_size)

        
        # 청크가 생성되지 않은 경우 오류 처리
        if not audio_chunks:
            print("오류: 오디오 청크가 생성되지 않았습니다.")
            return {"error": "오디오 생성 실패. 입력 텍스트를 확인하세요."}, 500

        # 모든 청크를 하나로 합칩니다.
        final_audio = torch.cat(audio_chunks, dim=-1)
        
        # 임시 파일 경로에 최종 오디오를 저장합니다.
        ta.save(output_path, final_audio, model.sr)
        print(f"오디오 파일이 임시 경로에 저장됨: {output_path}")

        # 응답이 전송된 *후에* 임시 파일을 삭제하도록 백그라운드 작업을 추가합니다.
        background_tasks.add_task(os.remove, output_path)

        # 생성된 오디오 파일을 클라이언트에게 반환합니다.
        return FileResponse(
            output_path, 
            media_type="audio/wav", 
            filename="generated_speech.wav" # 클라이언트가 다운로드할 때 보게 될 파일명
        )

    except Exception as e:
        print(f"심각한 오류 발생: {str(e)}")
        # 오류가 발생하면 임시 파일을 정리합니다.
        if output_path and os.path.exists(output_path):
            os.remove(output_path)
        return {"error": f"내부 서버 오류: {str(e)}"}, 500

@app.get("/")
def read_root():
    return {"message": "Chatterbox Multilingual TTS API가 실행 중입니다."}