# farm_insects 내의 각 클래스 나누기

# 생명주기 단계
candidate_stage = ["adult", "larva", "pupa", "egg"]  # 빈 문자열도 추가하여 단계 없이도 분류 가능

# 해충 클래스들
candidate_labels = [
    'Western Corn Rootworms', 
    'Citrus Canker', 
    'Fruit Flies', 
    'Africanized Honey Bees (Killer Bees)', 
    'Aphids', 
    'Colorado Potato Beetles', 
    'Thrips', 
    'Corn Earworms', 
    'Fall Armyworms', 
    'Corn Borers', 
    'Cabbage Loopers', 
    'Brown Marmorated Stink Bugs', 
    'Spider Mites', 
    'Tomato Hornworms', 
    'Armyworms'
]

# 텍스트 프롬프트 생성
text_prompts = []
for label in candidate_labels:
    for stage in candidate_stage:

        if stage:
            if label == 'Citrus Canker':
                text_prompts.append(f'a photo of {label}')

            else:
                text_prompts.append(f'a photo of {stage} {label}')
                
# text_prompts 중복제거
text_prompts = list(set(text_prompts))

print(f"총 {len(text_prompts)}개의 텍스트 프롬프트가 생성되었습니다.")
print("첫 5개 프롬프트 예시:")
for i, prompt in enumerate(text_prompts[:10]):
    print(f"{i+1}. {prompt}")
