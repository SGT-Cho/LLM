```markdown
# LLaMA-LoRA Fine-Tuning & WebUI Demo

본 레포지토리는 **LLaMA 3.1 8B** 모델에 대해 **LoRA(PEFT)** 방식으로 파인튜닝한 후,  
간단한 **챗봇(Web UI)** 를 구성하는 예시 과정을 담고 있습니다.

- **학습 데이터**: CSV 파일 예시 (음식이름, 음식설명 등)
- **파인튜닝 기법**: LoRA(PEFT)를 이용하여 GPU(MPS) 메모리를 절약
- **추론**: CLI(Python 스크립트) / Flask API + Web UI(HTML/JS)

---

## 1. 환경 준비

### 1.1 Conda(가상환경) 설치

```bash
# Conda가 이미 설치되어 있다면 스킵
# 예시로 miniconda 설치
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh
```

### 1.2 가상환경 생성 및 활성화

```bash
conda create -n llama-finetune python=3.9
conda activate llama-finetune
```

### 1.3 필수 라이브러리 설치

```bash
pip install torch torchvision torchaudio   # MPS(Apple Silicon) 지원 버전
pip install transformers accelerate peft
pip install datasets pandas flask flask-cors
```

> **주의**: Apple Silicon(M 시리즈)에서 GPU(MPS) 가속을 활용하려면,  
> PyTorch >= 2.0 버전이 설치되어 있어야 하며, `torch.backends.mps.is_available()`가 `True`인지 확인해 주세요.

---

## 2. CSV 데이터 준비

예시로 `food.csv` 파일이 다음과 같은 형식이라고 가정합니다:

```
음식이름,음식설명
기장밥,기장을 섞어 지은 밥으로, 고소한 맛과 영양이 풍부합니다.
곤드레밥,곤드레 나물을 넣어 지은 밥으로, 특유의 향과 부드러운 식감이 특징입니다.
...
```

- 동일 디렉토리에 `food.csv` 를 배치합니다.

---

## 3. LoRA 파인튜닝 스크립트

아래는 **로컬에 다운로드**된 **LLaMA 3.1 8B 모델**(예: `./llama-3.1-8b`)을 LoRA 방식으로 파인튜닝하는 예시 스크립트입니다.  
- 파일명: `lora_finetune.py`

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, TrainingArguments, Trainer)
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

def main():
    # -------------------------------------------------------
    # 1. 기본 설정
    # -------------------------------------------------------
    BASE_MODEL_PATH = "./llama-3.1-8b"            # 원본 LLaMA 모델 경로
    OUTPUT_DIR = "./lora-llama-finetuned"         # 파인튜닝 결과 저장 폴더
    CSV_FILE = "food.csv"                         # 예시 CSV
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # -------------------------------------------------------
    # 2. 모델 및 토크나이저 로드
    # -------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=False)
    # LLaMA 계열엔 pad_token이 없으므로, eos_token을 pad_token으로 설정
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        device_map="auto"  # or None -> base_model.to(device)
    )
    base_model.to(device)

    # -------------------------------------------------------
    # 3. 데이터셋 로드 & 전처리
    # -------------------------------------------------------
    dataset = load_dataset("csv", data_files=CSV_FILE)["train"]
    
    def tokenize_function(example):
        prompt = f"음식이름: {example['음식이름']}\n설명: "
        target = example['음식설명']
        input_text = prompt + target
        
        tokenized = tokenizer(
            input_text,
            truncation=True,
            max_length=256,
            padding="max_length"
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    tokenized_dataset = dataset.map(tokenize_function, batched=False)

    # -------------------------------------------------------
    # 4. LoRA 구성
    # -------------------------------------------------------
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],  # LLaMA 계열에서 주로 사용
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    peft_model = get_peft_model(base_model, lora_config)
    peft_model.print_trainable_parameters()

    # -------------------------------------------------------
    # 5. Trainer 설정 및 학습
    # -------------------------------------------------------
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=1,
        save_steps=50,
        logging_steps=10,
        evaluation_strategy="no",
        learning_rate=1e-4,
        fp16=False,
        bf16=False,
    )
    
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    trainer.train()
    
    # 학습 완료 후 LoRA 가중치 저장
    trainer.save_model()
    
    print("=== LoRA Fine-Tuning Completed ===")

if __name__ == "__main__":
    main()
```

### 실행

```bash
python lora_finetune.py
```

- 학습 후 `./lora-llama-finetuned` 폴더에 LoRA 가중치가 저장됩니다.

---

## 4. 추론 스크립트 (CLI)

학습이 끝난 모델을 로딩해 **CLI**에서 테스트할 수 있습니다.

- 파일명: `lora_inference.py`
```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    BASE_MODEL_PATH = "./llama-3.1-8b"
    LORA_ADAPTER_PATH = "./lora-llama-finetuned"

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        device_map="auto",
    )
    base_model.to(device)

    print("Loading LoRA adapter...")
    lora_model = PeftModel.from_pretrained(
        base_model,
        LORA_ADAPTER_PATH,
        device_map="auto"
    )
    lora_model.eval()
    lora_model.to(device)

    # 간단 테스트
    prompt = "음식이름: 곤드레밥\n설명: "
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = lora_model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            top_p=0.9,
            temperature=0.8
        )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("\n=== Inference Result ===")
    print(output_text)

if __name__ == "__main__":
    main()
```

- 실행:
  ```bash
  python lora_inference.py
  ```
- 모델이 곤드레밥에 대한 설명을 생성해 콘솔에 출력할 것입니다.

---

## 5. Flask API + Web UI 데모

### 5.1 Flask API 서버

- 예시 코드(`server.py`):

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

app = Flask(__name__)
CORS(app)  # CORS 허용

# ----- 전역 모델 로드 -----
BASE_MODEL_PATH = "./llama-3.1-8b"
LORA_ADAPTER_PATH = "./lora-llama-finetuned"
device = "mps" if torch.backends.mps.is_available() else "cpu"

print("[Server] Using device:", device)
print("[Server] Loading base model & tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    device_map="auto"
)
base_model.to(device)

print("[Server] Loading LoRA adapter weights...")
lora_model = PeftModel.from_pretrained(
    base_model,
    LORA_ADAPTER_PATH,
    device_map="auto"
)
lora_model.eval()
lora_model.to(device)


@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.get_json()
    user_input = data.get("user_input", "")
    prompt = f"사용자: {user_input}\nAI: "

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = lora_model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            top_p=0.9,
            temperature=0.8
        )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    answer = output_text.split("AI:")[-1].strip()

    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5052)
```

#### 실행

```bash
python server.py
```
- 이 상태로 5052 포트에서 API가 서빙됩니다.
- 브라우저나 HTTP 클라이언트에서 `POST http://<서버IP>:5052/api/generate` 로 `{ "user_input": "비빔밥에 대해 알려줘" }` 같은 JSON을 전송하면, 응답에 `{"answer": "..."} ` 형태가 돌아옵니다.

### 5.2 간단한 Web UI (HTML/JS)

- `index.html` (동일 디렉토리 혹은 로컬에서 접근 가능)
```html
<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8"/>
  <title>LLama-LoRA Chat</title>
  <style>
    body { font-family: sans-serif; max-width: 600px; margin: 20px auto; }
    #messages { white-space: pre-wrap; }
  </style>
</head>
<body>
<h1>LLama-LoRA Chat</h1>
<div id="messages"></div>
<input type="text" id="userInput" placeholder="질문을 입력하세요." style="width: 80%;"/>
<button id="sendBtn">전송</button>

<script>
  const messagesDiv = document.getElementById('messages');
  const userInput   = document.getElementById('userInput');
  const sendBtn     = document.getElementById('sendBtn');

  // Flask 서버 API URL
  // 서버가 192.168.0.6:5052 에서 실행된다면 아래와 같이:
  const API_URL = "http://192.168.0.6:5052/api/generate";

  sendBtn.addEventListener('click', async () => {
    const question = userInput.value.trim();
    if(!question) return;
    messagesDiv.textContent += "사용자: " + question + "\n";

    try {
      const resp = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_input: question })
      });
      const data = await resp.json();
      const answer = data.answer;
      messagesDiv.textContent += "AI: " + answer + "\n\n";
    } catch(err) {
      console.error(err);
      messagesDiv.textContent += "오류가 발생했습니다.\n\n";
    }

    userInput.value = "";
  });
</script>
</body>
</html>
```

- `file://.../index.html` 로 여는 경우 CORS 이슈가 있을 수 있으니,  
  간단히 `python -m http.server` 를 사용하거나, Flask static 폴더로 서빙하는 방법을 쓸 수도 있습니다.

---

## 6. 참고 사항

1. **Apple Silicon(M 시리즈) 성능**  
   - MPS 가속이 NVIDIA GPU만큼 빠르진 않을 수 있지만, 8B 모델 정도는 충분히 실험 가능합니다.  
2. **PEFT/Transformers 버전 충돌**  
   - LLaMA 계열 모델과 LoRA를 함께 사용 시, 버전 불일치로 `KeyError`가 발생하는 경우가 있습니다.  
   - `pip install --upgrade --force-reinstall transformers accelerate peft` 로 최신 버전을 사용해주세요.  
3. **CORS**  
   - 웹 브라우저에서 Flask API로 AJAX 요청 시, `flask-cors` 라이브러리로 CORS를 허용해야 합니다.  
4. **Chat 형태의 히스토리 유지**  
   - 더 정교한 대화를 위해서는, 이전 대화 내용을 prompt에 누적해서 전달하거나, LLaMA 시리즈의 대화 템플릿(예: `<s>`, `</s>` 토큰) 등을 활용할 수 있습니다.

---

## 7. 정리

- **CSV 기반 데이터**로 LoRA 파인튜닝 → **PEFT**로 가벼운 파라미터만 업데이트 → **64GB Unified Memory** 환경에서도 8B 모델 학습 가능  
- CLI 스크립트(`lora_inference.py`)나, **Flask API + Web UI**로 다양한 방식의 추론/챗봇 구현  
- 실제 서비스 단계에서는 **Gunicorn + Nginx**, **Docker** 등으로 프로덕션 환경을 구성하거나, 더 전문적인 MLOps 파이프라인을 적용할 수 있습니다.

이로써 **LLaMA + LoRA**를 이용한 **커스텀 파인튜닝**부터, 간단 **챗봇 UI** 구동까지 한 흐름을 마무리할 수 있습니다.
```
