import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
import os
import json
import csv
from datetime import datetime
from name import text_prompts

def load_model_and_processor():
    """모델과 프로세서를 로드합니다."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 중인 디바이스: {device}")
    
    model = AutoModel.from_pretrained("openai/clip-vit-base-patch32", torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model = model.to(device)
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor, device

def get_image_files(folder_path):
    """폴더별로 모든 파일을 찾습니다."""
    folder_files = {}
    
    for root, dirs, files in os.walk(folder_path):
        folder_name = os.path.basename(root)
        if folder_name != "farm_insects":  # 루트 폴더 제외
            folder_files[folder_name] = []
            for file in files:
                folder_files[folder_name].append(os.path.join(root, file))
    
    # 각 폴더 내 파일들을 자연스러운 숫자 순서로 정렬
    def natural_sort_key(filepath):
        import re
        filename = os.path.basename(filepath)
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', filename)]
    
    for folder_name in folder_files:
        folder_files[folder_name].sort(key=natural_sort_key)
    
    return folder_files

def classify_image(image_path, model, processor, text_prompts, device):
    """이미지를 분류합니다."""
    try:
        # 이미지 로드
        image = Image.open(image_path)
        
        # 텍스트와 이미지 처리
        inputs = processor(text=text_prompts, images=image, return_tensors="pt", padding=True)
        
        # GPU로 이동
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 모델 추론
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            most_likely_idx = probs.argmax(dim=1).item()
            most_likely_label = text_prompts[most_likely_idx]
            confidence = probs[0][most_likely_idx].item()
        
        return most_likely_label, confidence
    
    except Exception as e:
        print(f"이미지 처리 중 오류 발생: {image_path}, 오류: {e}")
        return None, 0.0

def extract_class_from_label(label):
    """예측 결과에서 클래스명만 추출합니다."""
    # "a photo of adult Africanized Honey Bees (Killer Bees)" 형태에서 클래스명 추출
    if "a photo of" in label:
        # "a photo of" 이후의 모든 텍스트가 클래스명
        parts = label.split("a photo of ")
        if len(parts) > 1:
            class_part = parts[1]
            # 성장단계 제거 (adult, larva, pupa, egg)
            growth_stages = ["adult ", "larva ", "pupa ", "egg "]
            for stage in growth_stages:
                if class_part.startswith(stage):
                    class_part = class_part[len(stage):]
                    break
            return class_part
    return label

def save_results(class_results, total_correct, total_files, overall_accuracy):
    """결과를 다양한 형식으로 저장합니다."""
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    # 전체 평균 신뢰도 계산
    all_confidences = []
    for class_name, results in class_results.items():
        for result in results:
            all_confidences.append(result['confidence'])
    overall_avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
    
    # 결과 저장 폴더 생성
    results_dir = f"results"
    os.makedirs(results_dir, exist_ok=True)
    print(f"📁 결과 저장 폴더 생성: {results_dir}")
    
    # 1. JSON 파일로 상세 결과 저장
    json_filename = os.path.join(results_dir, f"detailed_results.json")
    detailed_results = {
        'summary': {
            'total_files': total_files,
            'total_correct': total_correct,
            'overall_accuracy': overall_accuracy,
            'timestamp': timestamp
        },
        'class_results': class_results
    }
    
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)
    print(f"📄 상세 결과를 {json_filename}에 저장했습니다.")
    
    # 2. CSV 파일로 개별 이미지 결과 저장
    csv_filename = os.path.join(results_dir, f"individual_results.csv")
    with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['클래스', '파일명', '예측_라벨', '예측_클래스', '신뢰도', '정확_여부'])
        
        for class_name, results in class_results.items():
            for result in results:
                writer.writerow([
                    class_name,
                    result['file'],
                    result['predicted_label'] or '실패',
                    result['predicted_class'] or '실패',
                    result['confidence'],
                    '정확' if result['is_correct'] else '오류'
                ])
    print(f"📊 개별 결과를 {csv_filename}에 저장했습니다.")
    
    # 3. 요약 통계 CSV 파일
    summary_filename = os.path.join(results_dir, f"summary_statistics.csv")
    with open(summary_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['클래스', '총_이미지수', '정확_개수', '정확도(%)', '평균_신뢰도'])
        
        for class_name, results in class_results.items():
            correct_count = sum(1 for r in results if r['is_correct'])
            accuracy = (correct_count / len(results)) * 100 if results else 0
            avg_confidence = sum(r['confidence'] for r in results) / len(results) if results else 0
            
            writer.writerow([
                class_name,
                len(results),
                correct_count,
                f"{accuracy:.1f}",
                f"{avg_confidence:.3f}"
            ])
        
        # 전체 요약
        writer.writerow([])
        writer.writerow(['전체', total_files, total_correct, f"{overall_accuracy:.1f}", f"{overall_avg_confidence:.3f}"])
    print(f"📈 요약 통계를 {summary_filename}에 저장했습니다.")
    
    # 4. 텍스트 요약 파일
    text_filename = os.path.join(results_dir, f"results_summary.txt")
    with open(text_filename, 'w', encoding='utf-8') as f:
        f.write("제로샷 분류 결과 요약\n")
        f.write("=" * 50 + "\n")
        f.write(f"실행 시간: {timestamp}\n")
        f.write(f"전체 정확도: {total_correct}/{total_files} ({overall_accuracy:.1f}%)\n")
        f.write(f"전체 평균 신뢰도: {overall_avg_confidence:.3f}\n\n")
        
        for class_name, results in class_results.items():
            correct_count = sum(1 for r in results if r['is_correct'])
            accuracy = (correct_count / len(results)) * 100 if results else 0
            avg_confidence = sum(r['confidence'] for r in results) / len(results) if results else 0
            
            f.write(f"📁 {class_name}\n")
            f.write(f"   총 이미지 수: {len(results)}개\n")
            f.write(f"   정확도: {correct_count}/{len(results)} ({accuracy:.1f}%)\n")
            f.write(f"   평균 신뢰도: {avg_confidence:.3f}\n\n")
    
    print(f"📝 텍스트 요약을 {text_filename}에 저장했습니다.")
    
    # 5. README 파일 생성
    readme_filename = os.path.join(results_dir, "README.md")
    with open(readme_filename, 'w', encoding='utf-8') as f:
        f.write("# 제로샷 분류 결과\n\n")
        f.write(f"**실행 시간**: {timestamp}\n\n")
        f.write(f"**전체 정확도**: {total_correct}/{total_files} ({overall_accuracy:.1f}%)\n")
        f.write(f"**전체 평균 신뢰도**: {overall_avg_confidence:.3f}\n\n")
        f.write("## 파일 설명\n\n")
        f.write("- `detailed_results.json`: 모든 결과의 상세 JSON 데이터\n")
        f.write("- `individual_results.csv`: 개별 이미지별 분류 결과\n")
        f.write("- `summary_statistics.csv`: 클래스별 통계 요약\n")
        f.write("- `results_summary.txt`: 텍스트 형태의 결과 요약\n")
        f.write("- `README.md`: 이 파일\n\n")
        f.write("## 클래스별 성능\n\n")
        f.write("| 클래스 | 총 이미지 | 정확 개수 | 정확도 | 평균 신뢰도 |\n")
        f.write("|--------|-----------|-----------|--------|-------------|\n")
        
        for class_name, results in class_results.items():
            correct_count = sum(1 for r in results if r['is_correct'])
            accuracy = (correct_count / len(results)) * 100 if results else 0
            avg_confidence = sum(r['confidence'] for r in results) / len(results) if results else 0
            f.write(f"| {class_name} | {len(results)} | {correct_count} | {accuracy:.1f}% | {avg_confidence:.3f} |\n")
    
    print(f"📖 README 파일을 {readme_filename}에 생성했습니다.")
    print(f"\n💾 모든 결과 파일이 '{results_dir}' 폴더에 저장되었습니다!")

if __name__ == "__main__":
    # 폴더 경로 설정
    folder_path = "/home/shinds/my_document/pest/farm_insects"
    
    # 모델과 프로세서 로드 (한 번만)
    print("모델과 프로세서를 로드하는 중...")
    model, processor, device = load_model_and_processor()
    
    # 폴더별 이미지 파일들 찾기
    print("이미지 파일들을 찾는 중...")
    folder_files = get_image_files(folder_path)
    
    total_files = sum(len(files) for files in folder_files.values())
    print(f"총 {len(folder_files)}개 폴더, {total_files}개의 이미지 파일을 찾았습니다.")
    
    # 결과 저장용 딕셔너리
    class_results = {}
    
    # 폴더별로 각각 처리
    processed_count = 0
    for folder_name, image_files in folder_files.items():
        print(f"\n=== {folder_name} 폴더 처리 중 ===")
        class_results[folder_name] = []
        
        for i, image_path in enumerate(image_files):
            processed_count += 1
            print(f"처리 중: {processed_count}/{total_files} - {os.path.basename(image_path)}")
            
            label, confidence = classify_image(image_path, model, processor, text_prompts, device)
            
            if label:
                # 예측 결과에서 클래스명만 추출
                predicted_class = extract_class_from_label(label)
                is_correct = predicted_class.lower() == folder_name.lower()
                
                print(f"실제 클래스: {folder_name}")
                print(f"예측 결과: {label}")
                print(f"추출된 클래스: {predicted_class}")
                print(f"신뢰도: {confidence:.3f}")
                print(f"결과: {'✅ 정확' if is_correct else '❌ 오류'}")
                
                # 결과 저장
                class_results[folder_name].append({
                    'file': os.path.basename(image_path),
                    'predicted_label': label,
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'is_correct': is_correct
                })
            else:
                print("분류 실패")
                class_results[folder_name].append({
                    'file': os.path.basename(image_path),
                    'predicted_label': None,
                    'predicted_class': None,
                    'confidence': 0.0,
                    'is_correct': False
                })
    
    # 클래스별 결과 요약 출력
    print("\n" + "="*50)
    print("클래스별 분류 결과 요약")
    print("="*50)
    
    total_correct = 0
    total_files = 0
    
    for class_name, results in class_results.items():
        print(f"\n📁 {class_name}")
        print(f"   총 이미지 수: {len(results)}개")
        
        # 정확도 계산
        correct_predictions = 0
        label_counts = {}
        
        for result in results:
            pred_label = result['predicted_label']
            
            # 이미 저장된 predicted_class 사용
            predicted_class = result['predicted_class']
            
            # 실제 클래스와 예측 클래스 비교
            is_correct = predicted_class.lower() == class_name.lower() if predicted_class else False
            if is_correct:
                correct_predictions += 1
            
            # 예측 결과 분포 계산 (클래스명만 사용)
            if predicted_class in label_counts:
                label_counts[predicted_class] += 1
            else:
                label_counts[predicted_class] = 1
        
        accuracy = (correct_predictions / len(results)) * 100 if results else 0
        total_correct += correct_predictions
        total_files += len(results)
        
        print(f"   ✅ 정확도: {correct_predictions}/{len(results)} ({accuracy:.1f}%)")
        
        print("   예측 결과 분포:")
        for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(results)) * 100
            # 실제 클래스와 일치하는지 표시
            is_match = label.lower() == class_name.lower()
            match_indicator = "✅" if is_match else "❌"
            print(f"     {match_indicator} {label}: {count}개 ({percentage:.1f}%)")
        
        # 평균 신뢰도
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        print(f"   평균 신뢰도: {avg_confidence:.3f}")
    
    # 전체 정확도
    overall_accuracy = (total_correct / total_files) * 100 if total_files > 0 else 0
    
    # 전체 평균 신뢰도 계산
    all_confidences = []
    for class_name, results in class_results.items():
        for result in results:
            all_confidences.append(result['confidence'])
    overall_avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
    
    print(f"\n" + "="*50)
    print(f"🎯 전체 정확도: {total_correct}/{total_files} ({overall_accuracy:.1f}%)")
    print(f"🎯 전체 평균 신뢰도: {overall_avg_confidence:.3f}")
    print("="*50)
    
    print("\n모든 이미지 처리 완료!")
    
    # 결과 저장
    save_results(class_results, total_correct, total_files, overall_accuracy)