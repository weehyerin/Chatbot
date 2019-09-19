# Chatbot

### 챗봇의 종류
1. 규칙 기반
2. 머신러닝을 활용한 유사도 기반
3. 규칙과 머신러닝을 섞은 하이브리드 기반
4. 특정 시나리오에서 동작 가능해지는 시나리오 기반

**이번에는 딥러닝 모델을 통한 챗봇 구현 예정**
- sequence to sequence를 사용해 챗봇 제작

**챗봇을 위한 데이터가 거의 없음...**

데이터 : 송영숙 님이 제공한 Chatbot_data_for_korean

-------

## 데이터 분석
Q : 질문
A : 질문에 대한 답변
label : 0 - 일상 대화, 1 - 긍정, 2 - 부정
![image](https://user-images.githubusercontent.com/37536415/65215234-c6da3300-dae7-11e9-80b3-c58f942cc1b3.png)

### 1. 문장 길이 분석
**문장 전체에 대한 분석**
- 질문과 답변 모두에 대해 길이를 분석하기 위해 두개을 하나의 리스트로 만들기
1. 길이 분석하기(아래 세 가지 기준으로 분석)
    - 문장 단위의 길이 분석(음절)
    - 단어 단위의 길이 분석(어절)
    - 형태소 단위의 길이 분석
        - 음절 : 문자 하나하나
        - 어절 : 띄어쓰기 기준
        - 형태소 : 의미를 가지는 최소 단위
![image](https://user-images.githubusercontent.com/37536415/65215231-c2ae1580-dae7-11e9-937c-de457bf9e8a5.png)
    > 빨간색 : 어절 단위
    > 파란색 : 음절
    > 초록색 : 어절
    > 형태소나 어절의 경우 30, 45 길이에서 이상치 존재
    > **길이에 대한 분포를 통해 입력 문장의 길이를 어떻게 설정할 지 정의**
    > 결과 : 문장 길이 5 ~ 15 길이 중심으로 분포됨.

**질문, 답변 각각에 대한 분석**
- 위와 똑같은 방식으로 분석하면 됨. 

- 답변 데이터의 길이가 질문 데이터 보다 길다. 
- 문장의 최대 길이 정하기 : 긴 문장이 잘리지 않도록 25 정도로 잡음

### 2. 데이터 어휘 빈도 분석

`okt.pos('오늘밤은유난히덥구나')`
[('오늘밤', 'Noun'), ('은', 'Josa'), ('유난히', 'Adverb'), ('덥구나', 'Adjective')]
- 명사, 형용사, 동사만 사용할 예정 - 불용어 처리
    - 질문, 답변 데이터에서 명사, 형용사, 동사를 제외한 단어를 제거하기
~~~
for s in query_sentences:
    for token, tag in okt.pos(s.replace(' ', '')):
        if tag == 'Noun' or tag == 'Verb' or tag == 'Adjective':
            query_NVA_token_sentences.append(token)

for s in answer_sentences:
    temp_token_bucket = list()
    for token, tag in okt.pos(s.replace(' ', '')):
        if tag == 'Noun' or tag == 'Verb' or tag == 'Adjective':
            answer_NVA_token_sentences.append(token)
            
query_NVA_token_sentences = ' '.join(query_NVA_token_sentences)
answer_NVA_token_sentences = ' '.join(answer_NVA_token_sentences)
~~~

#### 질문 어휘 빈도에 대한 워드클라우드 그리기
1. 질문
![image](https://user-images.githubusercontent.com/37536415/65215425-6c8da200-dae8-11e9-8734-a6155b471377.png)
2. 답변
![image](https://user-images.githubusercontent.com/37536415/65215432-73b4b000-dae8-11e9-94d6-c44db37e72af.png)

**분석한 결과를 토대로 데이터를 전처리하고 모델을 만들기**

----------

## 시퀀스 투 시퀀스 모델링
### 모델 소개
- 시퀀스 형태의 입력값을 시퀀스 형태의 출력
- 하나의 텍스트 문장이 입력으로 들어오면 하나의 텍스트 문장을 출력
- 활용 : 기계번역, 텍스트 요약, 이미지 설명, 챗봇 등으로 활용
- RNN 모델 기반, 인코더 / 디코더로 나뉨(디코더에서 벡터를 활용해 재귀적으로 만들어내는 구조) 
    - 인코더 : 입력값을 받아 입력값의 정보를 담은 벡터 만들기
    - 디코더 : 벡터를 활용해 재귀적으로 출력값을 만들기
    ![image](https://user-images.githubusercontent.com/37536415/65215918-3c470300-daea-11e9-9fbc-8ea2162b12ee.png)
    - 아래 쪽이 인코더 인데 각 rnn step 마다 입력값이 들어감(입력값 : 하나의 단어), 인코더를 통해 하나의 벡터 값이 나옴(인코더의 정보를 요약해 담고 있는 벡터 - rnn의 마지막 은닉 상태 벡터값)
    - 인코더의 벡터가 디코더로 들어가며 새롭게 rnn 시작 --> 각 step 마다 하나씩 출력값이 나온다. 출력 역시 하나의 단어가 될 예정
    - 디코더 부분의 그림을 보면 각 스텝에서의 출력값이 다시 다음 스텝으로 들어감 : 출력 - 하나의 단어
![image](https://user-images.githubusercontent.com/37536415/65226681-3eb55700-db02-11e9-849f-c0af4a34e0d5.png)
> 파란색: 인코더 / 초록색 : 디코더
> 각 node마다 단어가 하나씩 들어감
> 임베딩 된 벡터로 바뀐 후 입력값으로 사용
> rnn의 경우 고정된 문장의 길이 정해야 함. 그래서 고정 4를 정해주고, 2개 글자만 사용한 것은 padding 처리해줌

----------

## 시퀀스 투 시퀀스 모델구현

> **데이터 살펴보기**
> 크게 2개의 폴더와 5개의 파일로 구성
> - 폴더 : 입력되는 데이터 / 출력되는 데이터
> - configs.py : 설정값
> - data.py : 데이터를 불러오고 가공하기
> - model.py : 시퀀스 투 시퀀스 모델 구현
> - main.py : 전체적으로 데이터를 불러오고, 모델을 실행하는 파일
> - predict.py : 모든 학습을 끝낸 모델을 사용해 챗봇 기능을 실제로 사용하기 위한 코드

### 1. configs.py
- tf.app.flags : 하이퍼파라미터 값 조정 ; 자료형에 따라 DEFINE_(data type) 형태로 지정

### 2. data.py
2-1. 데이터 전처리
~~~
FILTERS = "([~.,!?\"':;)(])"
PAD = "<PADDING>"
STD = "<START>"
END = "<END>"
UNK = "<UNKNOWN>"
MARKER = [PAD, STD, END, UNK]
CHANGE_FILTER = re.compile(FILTERS)
~~~
- 정규식에서 사용할 필터(미리 compile 해둠), padding, start, end, unknown 토큰과 해당 토큰들의 인덱스 값 지정

2-2. tokenizing
`def prepro_like_morphlized(data):` : 형태소를 분리, 형태소 기준으로 tokenizing
> 형태소 기준으로 토큰화 한 것을 띄어쓰기을 기준으로 다시 문장으로 만듦

2-3. 인코딩과 디코딩 부분에 각각 전처리
**인코더**
`def enc_processing(value, dictionary):`
> value : 전처리할 데이터
> dictionary : 단어 사전
    1. 정규 표현식 라이브러리를 이용해서 특수문자 제거
    2. 각 단어를 단어 사전을 이용해 단어 인덱스로 바꿈(어떤 단어가 단어사전에 없다면 UNK 토큰을 넣기)
    3. 문장이 너무 길면 문장을 자르고, 문장이 너무 짧으면 문장 padding 처리
        - return 값 : 전처리한 데이터, 패딩 처리하기 전 문장의 실제 길이
**디코더**
**두 가지 전처리 함수**
    1. 디코더의 입력으로 사용될 입력값을 만드는 전처리 함수 - <start>, 그래, 오랜만이야,<PADDING>
    2. 학습을 위해 필요한 라벨인 타깃값을 만드는 전처리 함수 - 그래, 오랜만이야, <END>, <PADDING>

1. 
`def dec_input_processing(value, dictionary):` 
- 시작 토큰을 넣기 + 나머지는 인코더와 비슷

2. 
`def dec_target_processing(value, dictionary):`
- 위의 1번과 다른 점은 <start> 토큰이 아니라 <end> 토큰을 넣기

2-3. 인덱스 벡터를 다시 문자열로 만들기
`def pred2string(value, dictionary):`

2-4. 데이터 tokenizing 하는 함수 - 단어 사전을 만들기 위해 
`def data_tokenizer(data):`

2-5. 단어 사전을 만드는 함수
`def load_vocabulary():`
- 2-4의 토크나이징으로 당어 리스트를 만든다. 
- 중복을 제거하고 단어 리스트 만들기(set 함수 사용)
- 특정 토큰 : <start>, <end> 등을 추가
- 저장

2-6. 단어 - 인덱스 vocabulary
`def make_vocabulary(vocabulary_list):`
- 단어에 대한 인덱스
- 인덱스에 대한 단어

2-7. 텐서플로우 모델에 데이터를 적용하기 위해 데이터 입력 함수 만들기
`def train_input_fn(train_input_enc, train_output_dec, train_target_dec, batch_size):`
- 입력 함수 parameter : 인코더에 적용될 값, 디코더에 적용될 값, 학습 시 디코더에 적용될 타깃 값
`def eval_input_fn(eval_input_enc, eval_output_dec, eval_target_dec, batch_size):`
- 위의 입력 함수와 다른 점 : repeat 함수의 인자값 - evaluation 때는 반복할 필요가 없으므로 한 번만

### 3. model.py
3-1. lstm
- 중간에 있는 신경망 : lstm 사용
`def make_lstm_cell(mode, hiddenSize, index):`
: 드롭아웃을 모듈화한 함수
- 현재 모델이 작동 중인 모드와lstm의 은닉 상태 벡터값의 차원, 여러 개의 lstm 스텝을 만기 때문에 각 스텝의 인덱스 값을 인자로 받음
- 드롭아웃 : 훈련할 때 임의의 뉴런을 골라 삭제하여 신호를 전달하지 않게 하는 것

3-2. 모델
`def model(features, labels, mode, params):`
parameter : feature : 모델 입력 함수를 통해 만들 feature 값(+ 인코더 입력, 디코더 입력), 디코더의 타깃, mode : 모델이 학습 상태인지, 검증 상태인지, 평가 상태인지, params는 모델에 적용되는 인자값

**인코더** : 입력값을 모델에 적용할 수 있게 벡터화, 임베딩 행렬을 초기화한 후 이 행렬을 통해 임베딩 벡터로 만들기
- 방법 2가지
1. 원-핫
2. 임베딩
임베딩을 사용하는 경우 임베딩 행렬 만들기 : 초기화(자비어(Xavier)) -> get_variable로 임베딩 행령 만들기

`tf.nn.embedding_lookup(params = embedding, ids = features['input'])` 각 단어를 임베딩 벡터로 만들기 
임베딩 : 단어를 벡터화 하는 것, 인덱스 정수가 아니라 실수화 하는 것
data.py의 인덱스화 된 것을 임베딩 화 시키는 과정

`with tf.variable_scope('encoder_scope', reuse=tf.AUTO_REUSE):` 인코더에 적용될 LSTM 신경망을 만드는 부분
- 인자로 전달되는 params의 멀티 층 사용 여부에 따라 두 가지 방법 중 하나로 구현
    - 멀티층을 사용하기 위해 `make_lstm_cell`을 여러 번 반복(층 수 만큼)해서 리스트로 만들기
- `dynamic_rnn` 함수에 적용(parameter : 위에서 임베딩한 입력값)

**디코더** : 디코더의 경우 LSTM의 펏 스텝의 은닉 상태 벡터 값을 인코더의 마지막 스텝의 은닉 상태 벡터 값으로 초기화 
- 기존의 논문의 lstm과는 다르게 구현함. 
- 디코더의 lstm 원래 : 각 스텝(t)의 결과가 다음 디코더 스텝(t+1)의 입력으로 사용
- 현 코드 : 디코더 입력값을 디코더 함수에 입력해서 만들어진 라벨값

**Dense**층 적용 : 결과값의 차원을 단어의 수만큼으로 변경하기 위해서

전체 모델이 끝나고 최종 출력물도 뽑았기 때문에 이제 남은것은 실제 라벨값과 비교하여 손실 값을 뽑은 후 모델을 학습을 모델에 학습

### 4. main.py

### 5. predict.py
- 고정된 question이 아니라, 사용자의 입력을 받아 예측한 문장 가져오기

-------------

## 트렌스포머 네트워크 모델
### 모델 소개
- seq 2 seq의 인코더 디코더 구조를 가지고 있지만 cnn, rnn을 기반으로 구성된 기존의 모델과 다르게 단순히 어텐션 구조만으로 전체 모델 만들기
- 기존의 seq 2 seq : '기분이 좋았어'라는 말을 하면 '기분이 좋다니 저도 좋아요'라고 문장을 생성한다고 하면, 순환 신경망의 경우 인코더에서 각 스텝을 거쳐 마지막 스텝에서 '기분이 좋다'라는 **문장의 맥락 정보가 반영되어 디코더에서 응답 문장을 생성**
    - 단순히 하나의 벡터에 인코더 부분에 해당하는 문장에 대한 모든 정보를 담고 있어 문장 안의 개별 단어와의 관계를 확인하기 어려움
    - 또한 문장 길이가 길수록 모든 정보를 하나의 벡터에 포함하기에는 부족
        - **문장이 길어지면 첫 번째 단어가 마지막 단어에 영향을 거의 주지 않음. 즉, 모든 단어의 정보가 잘 반영되고 각 단어 간의 관계를 잡기 어려움**


> 트렌스 포머 : 인코더 + 디코더
> 인코더에 입력한 문장정 보, 디코더에 입력한 문장 정보를 조합해서 디코더 문장 다음에 나올 단어에 대해 생성하는 방법
> 셀프 attention 기법 사용

### sel attention이란?
- 문장에서 각 단어끼리 얼마나 관계가 있는 지를 계산해서 반영
- 문장 안에서 단어들 간의 관계 측정 가능
- 각 단어들을 기준으로 다른 단어들과의 관계값을 계산

![image](https://user-images.githubusercontent.com/37536415/65248646-784f8780-db2d-11e9-8184-090244511f50.png)

각 단어에 대한 attention score 구하기
--> 어텐션 스코어 값을 하나의 테이블로 만든 것을 어텐션 맵이라 부름

![image](https://user-images.githubusercontent.com/37536415/65248953-04fa4580-db2e-11e9-887c-3663857b5822.png)
- 문장이 단어 벡터로 구성되어 있을 때, attention score를 구할 수 있어야 함.
![image](https://user-images.githubusercontent.com/37536415/65248962-075c9f80-db2e-11e9-8b62-afa606142261.png)

**문맥을 이해한 단어의 벡터**
![image](https://user-images.githubusercontent.com/37536415/65249197-7fc36080-db2e-11e9-8f89-c211e790ca9d.png)

--------
## 모델 구현
![image](https://user-images.githubusercontent.com/37536415/65249356-ce70fa80-db2e-11e9-890b-e820bd9eff3c.png)
인코더와 디코더에 들어가면 셀프 어텐션 기법을 활용해 해당 문장의 정보를 추출, 이 값을 토대로 디코더에서 출력 문장을 만듦