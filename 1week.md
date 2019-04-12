# 딥러닝을 이용한 자연어 처리


## Overview

머신러닝 
Data-Driven Algorithm

알고리즘은 어떠한 문제를 푸는데 있어 일련의 절차들이고, 머신러닝은 데이터를 기반으로 알고리즘을 설계하는 방법

프로그래머가 어떠한 문제를 전통적인 방법으로 해결하려고 한다면 다음과 같은 과정을 거친다 
1. 상세하게 문제를 정의
2. 문제를 해결하기 위해 알고리즘을 설계


머신러닝을 이용한 알고리즘 설계는 다음과 같은 과정을 따르게 됩니다.

1. 대략적인 문제의 정의를 제시합니다. (이것은 선택사항입니다.)
2. 학습할 수 있는 예시를 제공합니다.
3. 머신러닝 모델이 이 문제를 풀 수 있도록 학습을 시킵니다. (알고리즘을 만드는?)


### supervised learning 하려면 필요한것들
1. training example
2. loss function
3. validation and test example이 있고

### 결정해야할것들이 
1. hypotehsis sets (hypotehsis sets 안에있는 가설은 이문제를 풀기위한 알고리즘(모델?))
2. optimization algorithm 

hypothesis sets (H1 ... Hm) 을 결정해야하는데, 각각 hypothesis set 하나하나는 이 set 안에 모든 가능한 비슷한 모델들이 들어있는것
> 예를들어 첫번째 hypothesis set 안에는 SVM , 두번째 hypothesis set 안에는 같은 SVM인데 kernel function을 다른걸 쓴고 , 세번쨰는 KNN 같이 계속... 다른것들로

### 모든 것을 준비하고 결정하면

**지도학습은 가설에 대하여 최적화 알고리즘을 사용해 가장 좋은 모델을 결정**

1. **Training** : 각각의 hypothesis set안에서 가장 좋은 모델을 찾는다. ( hypothesis set안에 아주 많은 모델들이 있는데, 그 모델들 중에서 training set 으로 loss가 가장 작게나오는 모델을 찾는것)
    > 가령, SVM에서 Kernel function을 어떤 것을 쓸지, SVM의 정규화 계수 C를 어떻게 결정할지, Kernel function을 Gaussian Kernel function이라고 한다면, 거기서 bandwidth size가 어떻게 될 지에 따라서 SVM의 Hypothesis Set의 갯수는 다시 한 번 한없이 늘어나

2. **Model selection** : 이렇게 hypothesis set 하나하나 마다 모델들이 찾아졌으니까, 찾아진 모델중에서 뭐가 더좋은지 결정 (이 model selection 과정에서는 training set으로 하면 안됨(training set에 최적화된 모델들을 찾은것이니까) 그래서 validation set을 써서 loss 계산을 한다음에 그중에서 젤 좋은 하나를 고르는것)
	>hyper parameter optimization이 굉장히 중요한데, 이게 model selection 하는 단계이다. hyper paramter 를 매번 바꿀때마다 hypothesis set이 하나씩 계속 나오는것..

3. **Report**  : 마지막으로 Deploy 하기 전에 거치는 과정 
	>이 과정에서는 Test Set을 사용해서 가장 좋은 모델의 성능을 측정합니다.

즉 supervised learning이 하는일은 위에가 주어지고 결정하면, 최적의 알고리즘을 찾아주는것

### training , validation  ,test set
**초반에 시작할때 미리 trainig ,validation ,test set을 구분하는게 중요**
data set을 6 : 2 : 2 를 가장 많이 쓰는데, 이렇게 나누는 방법을 Simple Validation 이라고 하고, k-Fold Validation 이나 Leave-One-Out Validation 방법도 있다.


## Hypothesis Set
가설집합(Hypothesis set)은 매우 크고 많습니다. 머신러닝 접근 방법, 모델 구조, 하이퍼파라미터 등 요소를 하나씩 변화할 때마다 가설(Hypothesis) 하나가 세워지기 때문

### 어떤 방식의 머신러닝 접근 방법을 사용할 것인가?
* Classification: Support vector machines, Naive Bayes classifier, logistic regresstion …
* Regresstion: Support vector regression, Linear regression, Gaussian process …

### Hyperparameter는 어떤 것을 사용할 것인가?
* SVM을 사용한다면 kernel function은 어떤것을 사용하는가?
	* kernel func이 Gaussian kernel func이라면 거기서 Bandwith 사이즈는 어떻게 되는가?
* Neural network이라고 한다면 Convolution과 Recurrent 중에 어떤 것을 사용할 것인가?
	* Recurrent를 사용하면 unit 개수가 어떻게 되는가?

### Neural Net 에서의 Hypotehsis set 선택에서의 고려사항
NN에서 hypthesis 하나를 정하기 위해서는 일반적으로 2가지를 정함

1. Network Architecture 선택
> NN이라는게 작은 computing unit(node) 를 엮어서 그래프를 만들어서 input을 넣고 ouput이 계산이되서 나가는것인데 과연 그 그래프를 어떻게 만들것이냐..
2. Architecture 내의 파라미터 값
> 아키텍쳐가 주어지고 그 hypothesis안에서 서로 다른 모델들은 weight value , bias vector 등등의 paramter들이 어떻게 되어있는지의 차이 그러니 각 hypotehsis안에 infinitely many 하게 모델들이 들어있는것.. 그러니 모두 시도하기가 힘들어서 optimization 알고리즘을 쓰게되는것

![](https://github.com/wnsghek31/NLPusingdeeplearning/blob/master/image/dagnn.PNG)


![](https://github.com/wnsghek31/NLPusingdeeplearning/blob/master/image/daglogistic.PNG)
Logistic Regression을 도식화한것

즉 가설집합(Hypothesis set)은 무수히 많고, 이 중에서 가설집합(Hypothesis set) 하나를 결정하는 것은 비순환 그래프(DAG)를 만드는 것이라고 볼 수 있습니다. 그리고 그래프를 만들고 나면 그 안에 많은 Hyperparameter들이 있는데 여기서 Hyperparameter 값들을 결정하는 것이 학습

## Probability
**event set**안에 세상에서 일어날수있는 모든 가능한 event가 들어있는것
* Discrete : event < 무한대
* Continuous : event = 무한대

**random variable X**는 event set 안에있는 어떤 값도 취할수있다. 다만 그 값이 주어져있지않음.

**probability of event** 
확률[p(X =ei)] 은 event set에 속한 random variable에게 어떤값을 지정해주는 함수
- how likely would the i-th event happen?
- how often has the i-th event occur relative to the other events?

**Probability의 중요한 Properties**
1. Non-negative : p(X=ei) > 0
2. 모든 p(X=ei) 더하면 1이 되야댐

이것들은 당연한 사실이지만 loss function이 이 것들로부터 define 하게됨


### joint probability
p(Y=ej , X=ei) : How likely would ej and ei happen together?

### conditional probability 
p(Y=ej | X=ei) : Given ei , how likely would ej happen?
P(Y|X) = P(X,Y) / P(X)

### ==Margianl Probability==

### ==Marginalization==
joint distribution(두 개 이상의 확률 변수에 관계된 확률 분포)이 주어졌을 때(두 개가 동시에 일어날 확률이 주어졌을 떄) variable 하나에 대해서 관심이 없는 상태
> 예를들어 동전 두개를 던질때, 이 동전 두개가 독립적이지 않다고 하면
	첫번째 동전이 어떻게 나오느냐에 따라서 , 두번째 동전이 head이냐 tail이냐가 바뀐다고 했을때.. 어느순간 보니 첫번째 동전은 아무 의미가없는것 같다. 만약 두번째 동전이 head가 나온것이 중요하다고 생각하면 , 두번재 동전이 head가 나온 case에 대해서 첫번째 동전의 case를 모두 더한다. 그럼 이것이 marginalize라는것.
즉 여러개의 확률 변수로 구성된 조합 확률 분포 (joint distribution)에서 한가지 변수에 대한 확률값을 추정하기 위해 나머지 변수를 모두 적분하여 제거하느 과정을 말하는것

## Loss Function

모델을 학습시키기 위해서는 Optimization을 해야하고 , Optimization을 하기 위해서는 Loss Function을 정의해야함

이 강의에서는 확률 분포(distribution)에 따라서 자연스럽게 Loss function을 결정하는 방법에 대해서 설명


기존 지도학습은 input x 를 넣을때 output y가 어떤 값인지를 산출하는것이다. 하지만 좀만 다르게 생각해서, **input x를 넣을때 output y값이 y'일 확률을 구하는것으로 생각한다면 p(y=y'|x) 로 조건부확률**이다.
> 즉 기존의 지도학습 관점에선 확률이 1인 y'을 찾는 것이지만
새로운 관점에서는 input이 주어졌을때 output으로 될수 있는 value들은 굉장히 많을것이다.(즉 event set 에 있는 모든 가능한 것들을 말하는것.) 
그중에서 어떤것이 가장 likely 하고 어떤것이 unlikely 한지로 문제를 바꿔서 생각하자.

이렇게 바꾸고 문제들의 종류에 따라서 distribution을 어떤걸 쓸지를 결정을 해줘야댐

1. 이진 분류(Binary Classification) → 베르누이(Bernoulli) 분포
	> postive class인지, negative class인지 분류하는 문제 

2. 다중 분류(Muticlass Classification) → 카테고리(Categorical) 분포
	> 이미지들을 input으로 받아서 사람인지, 차인지, 동물인지를 판단하는 문제
3. 선형 회귀(Linear Regression) → 가우시안(Gaussian) 분포
	>  regression이니 output은 infinitely many . linear regression 같은 경우 , x가 주어졌을때 y값들이 gaussian distrbution을 따르는데, mean 과 covariance가 주어지면 두개의 차이가 얼마나 큰지를 보고 , 차가 크면 probability가 낮아지고  가까우면 가까울수록 probability가 높아져.
4. 다항 회귀(Multimodal Linear Regression) → 가우시안 믹스쳐(Mixture of Gaussians)

**인공신경망 모델의 아웃풋이 조건부 확률 분포라고 한다면 이를 사용해서 비용함수를 정의 한다면, 모델이 출력한 조건부 확률분포(Conditional distribution)가 훈련 샘플의 확률분포와 최대한 같게 만드는 것입니다. 즉, 모든 훈련 샘플이 나올 확률을 최대화 하는 것**

![](https://github.com/wnsghek31/NLPusingdeeplearning/blob/master/image/MaximumLikelihoodEstimation.PNG)

이 관점에서는 DAG가 어떤 아키텍처였는지, 모델(pθ)이 어떻게 생겼는지, 무슨 계산을하는지에 대해서는 전혀 신경 쓸 필요가 없습니다. 그저 확률이 output인 순간. 확률분포가 나오는 순간?), 공식을 적용해서 loss function을 만들 수 있습니다.(classifier 건 regression 모델이건) 이를 **최대 우도 측정(Maximum Likelihood Estimator)**라고한다


![](https://github.com/wnsghek31/NLPusingdeeplearning/blob/master/image/lossfunc.PNG)

최종적으로, 최대 우도 추정(Maximum Likelihood Estimator)은 모든 훈련 샘플이 나올 확률을 최대화하는 것이고, 우리의 목적은 Loss Fuction을 만들고 그것을 쵯최소화 하는것이기에 마이너스(-) 를 붙힘.

위에서 hypothesis set을 어떻게 만드는지를 본것 (DAG를 만든것) 
만들었을때 각 파라미터들이 hypothesis set 내의 모델들을 결정을 해준다.
그렇게 Neural Net을 만들었을때, output은 하나의 값이 아니라, Distribution이 output이 되게 한다면,
Loss function도 negative log probability를 계산하므로서 자동으로 나온다
지금까지 인공신경망(Neural network)을 만드는 방법, 음의 로그확률(Negative Log-probability)를 정의하는 방법 그리고 이 두 개를 하나의 비순환 그래프(DAG)로 표현하는 방법을 배웠습니다.

1. arbitrary architecture로 Neural Net을 만든다
2. negative log probability로 per-example loss function을 정의
3. 위의 두 가지를 포함하는 하나의 DAG를 정의



## OPTIMIZE

비용함수(Loss Function)를 구하는 방법을 알았으니 비용함수를 이용하여 최적화(Optimization)를 해야함

1. 한 점(Point)을 하나 고름
2. 선택한 점을 움직여서 (파라미터를 바꿔서) 비용을 낮춤
3. 2번 계속 반복
4. 더이상 비용 낮출수 없는 점(파라미터) 를 찾음

관건은 점을 어떻게 움직여야 하는지에 있다. 다음은 점을 움직이는 2가지 방법

### Local, Iterative Optimization: Random Guided Search
**장점** : loss function , DAG 가 어떤것이든지 바로 쓸수있다
**단점** : 단점은 차원이 작으면 잘 되지만 높은 차원을 가지게 되면 계산이 안된다고 하지만, 요즘음 클러스터(data)가 크면 다 된다. low dimension에서는 잘 되지만 , high dimension에서는 efficiency가 떨어짐


1. 랜덤하게 처음 시작 하나 뽑음
2. 노이즈 막 넣고 10개정도 찾은다음에 가장 낮은애로 감
3. 2번 반복해서 궁극적으로 젤 작은애로 간다.


![](https://github.com/wnsghek31/NLPusingdeeplearning/blob/master/image/randomguid.PNG)

> 랜덤하게 처음 시작 하나 뽑아서하고 , 노이즈 막 넣고 10개정도 찾은다음에 가장 낮은애로 가고(1) , 노이즈를 막넣어서 또 가장 낮은애 하니까 이번엔 잘안되서 좀 올라가고 (2) , 막 계속 반복하는거 .. 궁극적으로 낮은데로 간다. 

미분계산식이 안될때 미분하는 방법이 , 살짝움직이는걸 여러개 뽑아보고 , 쪼금씩 움직이는걸 보고 평균을 구해서 미분하듯이 만들수있다. 이 이론이 random guide search


그래서 !! loss function 과 DAG에 있는 모든 computation node들이 continuous 하고 differentiable 하다면  (모두 완전히 까지는아니지만 almost 하면) 효율적인 방법이 있다

### Gradient-based optimization
장점: Random Guided search 에 비해서 탐색영역은 작지만 확실한 방향은 정할 수 있습니다
단점 : 학습률(Learning Rate)이 너무 크거나 작으면 최적의 값으로 못갈 수도 있습니다
한점에서 loss function의 gradient가 뭔지 , 어느 방향으로 파라미터를 움직여야 값이 작아질지

1. 가설에서 랜덤한 점을 선택합니다.
2. 위에서 선택한 점에 미분을 하여 기울기를 구합니다.
3. 기울기를 가지고 방향을 결정하고 점을 학습률(Learning Rate) 만큼 이동시켜 비용을 낮춥니다.
4. 최적의 비용이 나올 때까지 1~3을 반복합니다.

![](https://github.com/wnsghek31/NLPusingdeeplearning/blob/master/image/gradientdescent.PNG)
