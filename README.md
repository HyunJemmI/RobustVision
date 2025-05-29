# Contrastive Learning을 활용한 Light-Noise Robust Traing Method

SimCLR 기반 Flow-matching idea 사용

# Flow
1. Light-Noise가 쉽게 존재할 수 있는 환경에 대한 dataset을 기반으로 (BDD100K) Noise 증강(Bloom, Glare, Gamma) 
2. contrastive learning을 통한 noise 환경에서도 같은 image feature 추출 
3. downstream(TuSimple dataset 사용) : lane-detection task를 수행하도록 fine tuning
4. noise augmentation 2 (TuSimple)
5. downstream test → noise augmentation 2 dataset(noise augmented TuSimple)
6. Tusimple dataset에서 → noise 추가한 img와 원본 사진 간 cosine similarity를 비교
7. 이후 유사도가 높은데도 높은 noise에서 사진을 제대로 검출하는지 비교 (벤치마킹)

# GPT-based Model Feature
## **🧠 SimCLR 기반 Noise-Robust 모델 설계 요약**

### **1. ✅ 모델 구조 (ResNet18 + Projector)**

- **Backbone**: ResNet-18에서 AvgPool, FC layer 제거
    
    → Classification 대신 **feature representation 추출**에 집중
    
    → 출력: [B, 512, 7, 7]
    
- **AdaptiveAvgPool2d((1,1))**
    
    → 입력 해상도와 무관하게 항상 [B, 512, 1, 1]으로 압축
    
    → 이후 flatten()을 통해 [B, 512] feature 벡터 확보
    
- **Projector Head**:
    
    Linear(512→512) → BN → ReLU → Linear(512→256) → BN(no affine)
    
    → SimCLR 논문 기반 구성
    
    → Nonlinearity + 정규화 통한 학습 안정성 확보
    

---

### **2. 🔥 NT-Xent Loss 구성**

- **Cosine Similarity 기반** 유사도 계산
    
    → Positive pair 유사도는 높이고, Negative pair는 낮춤
    
- temperature (T) 파라미터 도입
    
    → 유사도 분포 sharpness 조절
    
    → 작을수록 모델이 더 민감하게 positive/negative 구분
    
    → 실험적으로 T=0.35 사용 (0.07~0.5 사이 권장)
    

---

### **3. 🧪 Input 및 Transform**

- 이미지에 **이미 noise augmentation이 적용된 상태**
    
    → 추가적인 transform은 제외
    
    → 단순히 Resize(224) + Normalize만 적용
    

---

### **4. ⚙️ 기타 학습 안정성 요소**

- **AMP(Auto Mixed Precision)**
    
    → NVIDIA 6000 ADA GPU 사용 시 자동 활성화
    
    → 연산 속도 및 메모리 효율 개선
    
- **Gradient Accumulation**
    
    → batch_size=256, accum_steps=2 → 유효 배치 크기 512
    
- **Collapse 경고 감지**
    
    → neg_cos 평균값이 특정 임계치(0.3) 이상일 경우 collapse 가능성 출력
    
- **Early Stopping**
    
    → validation loss 기준, best 모델 저장 및 조기 종료

**두 모델 학습 방식의 핵심 차이점**

1. **백본 초기화(Pre-training)**
    - **Baseline**: ResNet-18을 ImageNet 지도학습(pretrained) 가중치로 초기화 → DeepLabV3 전체(백본+헤드) fine-tune
    - **SimCLR**: ResNet-18을 SimCLR self-supervised 가중치로 초기화 → 동일한 DeepLabV3 헤드를 붙여 fine-tune
2. **전처리(Normalization) 파이프라인**
    - **Baseline**: A.Resize→A.Normalize(mean,std)→ToTensorV2
    - **SimCLR**: A.Resize→ToTensorV2 → 모델 입력 전에 .float().div(255.0)
3. **학습 스케줄**
    - **Baseline**: 백본과 헤드를 동일한 learning rate로 학습
    - **SimCLR**: 백본은 낮은 lr, 헤드는 높은 lr으로 두 단계 스케줄링