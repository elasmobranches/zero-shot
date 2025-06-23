## 2025년 AI로봇융합심화 과제 깃허브 코드입니다.

## DL_models에는 실험에 사용한 모델들의 ipynb 파일이 있습니다.


## name.py는 농작물 해충 이미지 분류를 위한 텍스트 프롬프트를 생성하는 파일입니다.

### 주요 기능:
- **생명주기 단계**: adult, larva, pupa, egg 단계별 분류
- **해충 클래스**: 15가지 주요 농작물 해충 정의
- **프롬프트 생성**: "a photo of [단계] [해충명]" 형태의 텍스트 프롬프트 자동 생성

## zeroshot.py는 CLIP 모델을 사용한 제로샷 이미지 분류를 수행하는 메인 파일입니다.

### 주요 기능:
- **모델 로드**: OpenAI CLIP ViT-Base-Patch32 모델 사용
- **이미지 분류**: 폴더별 이미지 자동 분류
- **결과 분석**: 클래스별 정확도 및 신뢰도 계산
- **결과 저장**: JSON, CSV, 텍스트 등 다양한 형식으로 결과 저장


### 사용 방법:
1. `zeroshot.py` 실행하여 이미지 분류 수행
2. `results/` 폴더에서 결과 확인

## Kaggle Notebook으로 간단하게 코드를 실행시켜볼 수 있습니다.
- 위의 코드와 똑같진 않습니다. 단순한 구현입니다.
- https://www.kaggle.com/code/elasmobranch/kaggle-pest-zeroshot 