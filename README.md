# covid-detect

### Capston Design 2 Project

> 컴퓨터공학과 2016104146 이승윤

## 연구 배경

2019년 부터 이어진 코로나 바이러스 감염증(COVID-19)은 현재까지도 확산 증세를 보이고 있다. 바이러스는 COVID-19 감염자와 인접한 거리에서 호흡할 때 바이러스를 들어마시게 되거나 오염된 표면에 접촉한 손으로 눈, 코 또는 입을 통해 2차적인 침입을 허용하게 될 경우 감염될 수 있다. 또한 바이러스는 실내와 붐비는 장소에서 보다 쉽게 전파된다. 따라서 비교적 쉬운 경로를 통해 감염될 수 있는 바이러스이기 때문에 감염자의 효과적인 색출(스크리닝)이 중요한 단계를 가지고 있다.
이에 따라서 본 프로젝트에서는 감염자의 효과적인 색출 접근법 중 하나인 흉부 방사선 촬영 영상을 활용한 검사를 통해 딥러닝을 사용한 COVID-19 판별 프로그램을 구현하고자 한다.
이를 통해 직접 모델 구축 및 테스트를 경험하고, 딥러닝 및 인공지능에 대한 이해를 높일 뿐 아니라 해당 모델을 통해 의료산업에서의 인공지능 활용 범위의 증대에 대한 전망을 가늠할 수 있을 것이다.

## 관련연구

### COVID-Net

COVID-Net은 COVID-19 선별을 위해 설계된 최초의 신경망 아키텍처로 경량 PEPX (projection-expansion-projection-extension) 설계를 도입하여 표현 능력을 향상시킴과 동시에 계산 복잡성을 크게 줄인다. github과 kaggle사이트에 업로드된 약 14000개의 흉부 엑스레이 데이터셋을 전처리함과 동시에, 전처리된 데이터셋을 통해 학습하며, 이를 통해 결론을 도출한다. COVID-Net은 종합적으로 93.3%의 테스트 정확도를 달성하였으며, 각 분류된 모델별 PPV(Positive Predicate Value)또한 모두 90%가 넘는 예측성공률을 보여주고 있다.

### Brixia

COVID-19 환자의 CXR 사진을 딥 네트워크를 통해 분석하여 중증도를 평가하는 프로젝트 및 데이터베이스이다. 이는 점차 확산되고 있는 COVID-19의 검증 및 의료 절차를 간소화하기 위해 질병의 진행을 모니터링하는 정확도와 여러 임상의 간의 조정 및 의사 소통 수준을 개선하기 위해 만들어진 프로젝트이다.
해당 프로젝트의 특징은 다음과 같다.

- CXR에 대한 폐렴 중증도 평가를 위해 종단 간 다중 네트워크 아키텍처를 설계
- 약 4703개의 CXR 영상, 그리고 해당 영상 이미지에 라벨을 첨가하여 전처리한 대규모 데이터 셋을 보유
- 5명의 전문 방사선과 의사가 해당 이미지에 코멘트(주석)를 단 세트를 제공

### ResNet

ResNet은 마이크로소프트사에서 개발한 알고리즘이며, 해당 알고리즘은 152개의 layer를 갖는다. 기존까지는 네트워크의 층(layer)을 더 쌓으며 깊은 네트워크를 구현하여 성능 향상을 이루고 있었다. 하지만 실제로 어느정도 이상 깊어진 네트워크는 vanishing이나 exploding gradient와 같은 문제들이 발생하여 오히려 성능이 떨어지는 경우가 발생하였다. 이러한 degradation problem을 해결하기 위해 제안되었다.

## 제안 연구의 중요성/독창성

해당 연구를 통해 흉부 X-Ray 영상을 통해 정상 분류/ 바이러스성 페렴 분류/ COVID-19의 감염 여부를 판별하는 프로그램을 구현하는 것이 최종 목표이다. 또한 해당 프로그램을 만들기 위해 위와 같은 여러 관련 자료들을 연구/분석함으로써 의학적 기술과 딥러닝과 같은 인공지능의 기술에 대한 이해를 높이고, 이를 활용하기 위한 지식을 쌓는 것을 목표로 한다.

## 연구 내용(방법 및 실험)

1. 프로젝트 사용 언어: Python

- 딥러닝을 구현하는데 있어 가장 보편적인 언어인 Python을 채택하였다. 또한 코드 구동 환경으로 Google의 GPU 서버와 코드에디터등을 제공해주는 Google Colab을 사용하였다. 이를 통해 로컬에서 코드를 작성/저장 및 구동하지 않고 웹 브라우저 상에서 코드를 작성하고 모델을 학습시킬 수 있다는 점에서 큰 장점이 되었다.

2. 사용 라이브러리: PyTorch

- 앞서 언급한 대로 COVID-Net 오픈소스는 tensorflow 라이브러리로 구성되어있다. 하지만 조금 더 직관적이고 진입 난이도가 쉬운 PyTorch 라이브러리를 사용하여 해당 프로젝트를 시행하였다.

3. layer: ResNet18

4. 데이터셋: Kaggle사이트 내에 존재하는 COVID-19 Radiography 데이터셋을 사용하였으며, 정상 흉부 이미지 1311장, 단순 바이러스 성 질환으로 판정받은 이미지 1315, Covid-19를 판정받은 흉부 x선 이미지 189장으로 되어 있다.

## 연구 결과
- 이미지 시각화 및 라벨링
색글씨별 라벨: 초록색인 경우, 판별 성공 / 빨간색인 경우, 판별 실패
<img width="628" alt="스크린샷 2022-12-04 오후 11 20 38" src="https://user-images.githubusercontent.com/77139957/205495899-876d92c0-50e4-444f-ab11-b4e67de52903.png">

- 초기 학습 없이 예측한 결과
예측 성공률이 매우 낮은 것을 확인할 수 있다.
<img width="646" alt="스크린샷 2022-12-04 오후 11 22 10" src="https://user-images.githubusercontent.com/77139957/205496016-4f3c3140-3e28-48a1-9131-bcd286ee302a.png">

- 학습 후 예측 결과
### 테스트 이미지
nomal: 30 images
viral: 30 images
covid: 30 images

|Validation Loss|Accuracy|
|------|---|
|0.1597|0.9556|

## Reference

[1] Wang, Linda, Zhong Qiu Lin, and Alexander Wong. "Covid-net: A tailored deep convolutional neural network design for detection of covid-19 cases from chest x-ray images." Scientific Reports 10, no. 1 (2020): 1-12.

[2] Covid-CXR:
https://www.kaggle.com/datasets/andyczhao/covidx-cxr2?select=competition_test

[3] https://github.com/ieee8023/covid-chestxray-dataset.git

[4] https://github.com/agchung/Figure1-COVID-chestxray-dataset.git

[5] https://github.com/agchung/Actualmed-COVID-chestxray-dataset.git

[6]https://github.com/IliasPap/COVIDNet.git

[7] Deng, J. et al. Imagenet: a large-scale hierarchical image database. In 2009 IEEE Conference on Computer Vision and Pattern Recognition 248–255 (IEEE, 2009).
[8] https://github.com/lindawangg/COVID-Net
[9] Brixia: https://brixia.github.io/

### Contact: email: syoon624@naver.com
