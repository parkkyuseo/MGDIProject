# 실험 스크립트

## 1. 설정

- `within-subject` 실험
- 참가자 1명당 `4개 condition`
- condition 순서는 `Latin square`로 counterbalancing
- counterbalancing 기준 순서는 `Macro Near -> Macro Side -> Micro Near -> Micro Side`
- 각 condition에는 `Placement`, `Rotation`, `Scaling`이 포함됨
- 현재 workflow에서는 각 phase마다 `5개 tool`을 사용함

`Participant ID`의 숫자 부분으로 sequence를 결정합니다.

## 2. 진행자 요약

각 참가자는 네 개의 condition을 모두 수행합니다.

각 condition에서:

- 참가자는 화면에 나온 자세를 맞춥니다
- 참가자는 condition 시작을 위해 `triple tap` 합니다
- 시스템은 task phase 전에 짧은 practice를 보여줄 수 있습니다
- 참가자는 `Placement`, `Rotation`, `Scaling`을 수행합니다
- 현재 workflow에서는 각 phase를 `5개 tool`로 수행합니다
- condition이 끝나면 짧은 설문을 진행합니다

참가자에게 꼭 전달할 핵심 규칙:

- `Triple tap`은 현재 시도를 제출하는 동작입니다
- `Macro`에서는 주로 팔과 손의 큰 움직임을 사용합니다
- `Micro`에서는 팔을 최대한 고정하고 phone swipe로 조정합니다
- condition 시작을 위해 `triple tap` 한 직후에는 시스템이 잠시 phone connection과 stillness를 확인합니다

## 3. 세션 시작 전

### 진행자 체크리스트

- HMD, phone, study computer를 준비합니다
- 동의서 준비
- 설문을 준비합니다. (컨디션별 설문 준비)
- 필요하면 보상 관련 문서 준비
- 참가자별 counterbalanced condition order 확인
- 숫자가 포함된 `Participant ID` 할당
- phone app 사용 가능 여부 및 QR workspace setup 확인
- basket과 tool 위치 확인
- 세션이 participant ID 입력 화면에서 시작되는지 확인

### 권장 사전 확인

참가자 도착 전 다음을 확인합니다.

- 시작 화면에서 `Participant ID`를 입력할 수 있는지
- QR 스캔 후 runtime이 시작되는지
- phone connection이 정상인지
- 자세 예시 이미지가 잘 보이는지
- phone에서 `triple tap`이 정상 인식되는지

## 4. 환영 및 동의

### 진행자

- 참가자를 맞이합니다
- 이름과 예약을 확인합니다
- 진행이 괜찮은지 확인합니다
- 동의서를 제공합니다

### 읽어줄 문구

> 안녕하세요. 오늘 실험에 참여해 주셔서 감사합니다.
>
> 오늘 실험에서는 mixed reality 시스템과 phone 기반 입력 방식을 사용해서 물체 조작 과제를 수행하게 됩니다.
>
> 시작하기 전에 이 동의서를 천천히 읽어 주세요.
>
> 질문이 있으면 언제든지 말씀해 주세요.
>
> 참여에 동의하시면 서명해 주세요.

### 동의 후

### 읽어줄 문구

> 감사합니다.
>
> 다음으로 간단한 배경 설문을 부탁드리겠습니다.

## 5. 인구통계 설문

### 진행자

- 인구통계 설문을 제공합니다
- 참가자가 완료할 때까지 기다립니다
- 절차 관련 질문에는 답하되, 실험 가설은 설명하지 않습니다

### 읽어줄 문구

> 이 짧은 배경 설문을 작성해 주세요.
>
> 다 작성하시면 말씀해 주세요.

## 6. 실험 개요 설명

### 읽어줄 문구

> 이 실험에서는 같은 종류의 task를 여러 condition에서 수행하게 됩니다.
>
> condition에 따라 제어 방식과 손을 두는 위치가 달라집니다.
>
> 모든 condition에서 목표는 target을 최대한 정확하게 맞추는 것입니다.
>
> 준비가 되면 `triple tap`으로 현재 시도를 제출하시면 됩니다.
>
> 이 실험은 within-subject 방식이기 때문에, 모든 condition을 직접 수행하게 됩니다.
>
> condition 순서는 참가자마다 counterbalancing됩니다.

### 짧게 설명할 때

> 여러 condition에서 같은 task를 하게 됩니다.
>
> 매번 최대한 정확하게 맞춰 주세요.
>
> 현재 시도를 제출할 준비가 되면 `triple tap` 하시면 됩니다.

## 7. 기기 세팅

### 진행자

- app을 켜고 할당된 `Participant ID`를 입력합니다
- 참가자에게 HoloLens를 전달합니다
- 그 다음 참가자에게 phone을 전달합니다
- phone을 편하게 잡을 수 있는지 확인합니다

### 읽어줄 문구

> 먼저 app을 켜고 participant ID를 입력하겠습니다.
>
> 그 다음 HoloLens를 전달하겠습니다.
>
> 그 다음 phone을 드리겠습니다.
>
> Phone은 편한 방식으로 잡아 주세요.

### 진행자 메모

시작 화면에서는:

- 제가 할당된 `Participant ID`를 입력합니다
- 그 다음 runtime scene으로 넘어갑니다
- runtime scene에는 `Scan the QR code to begin.` 문구가 표시됩니다

## 8. Runtime 시작

### 진행자

- runtime instruction이 나오면 참가자에게 QR 스캔을 요청합니다
- workspace alignment가 올바른지 확인합니다

### 읽어줄 문구

> 시작하려면 QR 코드를 스캔해 주세요.
>
> workspace 준비가 끝나면 condition을 시작하겠습니다.

## 9. 공통 task 규칙

### 읽어줄 문구

> 모든 task에서 제출하기 전에 target을 최대한 정확하게 맞춰 주세요.
>
> 각 task에는 별도의 시작 버튼이 없습니다. task가 시작되면 바로 수행하시면 됩니다.
>
> 현재 시도를 제출할 준비가 되면 `triple tap` 해 주세요.
>
> 짧은 practice가 나오면 화면 안내를 따라 하시면 됩니다. practice는 익숙해지기 위한 단계입니다.

### 읽어줄 문구

> condition 사이에는 짧은 휴식과 짧은 설문이 있습니다.
>
> 이상하거나 헷갈리거나 불편한 점이 있으면 바로 말씀해 주세요.

## 10. Condition 시작 안내

### 진행자

- 각 condition 전에 화면의 자세 이미지가 기준이라는 점을 다시 알려줍니다
- 참가자가 자세를 맞출 때 서두르게 하지 않습니다
- 참가자가 condition 시작을 위해 `triple tap` 한 뒤에는 시스템이 phone connection과 stillness를 확인할 시간을 줍니다

### 읽어줄 문구

> 각 condition이 시작될 때는 화면에 보이는 자세를 맞춰 주세요.
>
> 자세를 맞춘 뒤 `triple tap` 해서 시작해 주세요.
>
> 그 다음에는 시스템이 준비될 때까지 잠깐만 가만히 있어 주세요.

## 11. Condition별 설명 문구

각 condition 시작 전에 해당 설명을 읽어줍니다.

### 11.1 Macro Near

### 읽어줄 문구

> 이 condition에서는 머리 근처에서 macro control을 사용합니다.
>
> 화면에 나온 것처럼 손을 머리 근처에 두어 주세요.
>
> Macro control에서는 주로 팔과 손의 큰 움직임을 사용합니다.

### 11.2 Macro Side

### 읽어줄 문구

> 이 condition에서는 몸 옆에서 macro control을 사용합니다.
>
> 화면에 나온 것처럼 손을 몸 옆에 두어 주세요.
>
> Macro control에서는 주로 팔과 손의 큰 움직임을 사용합니다.
>
> 이 side condition에서는 움직임 매핑이 바뀝니다.
>
> 위아래 움직임과 앞뒤 움직임이 서로 바뀌어 매핑됩니다.

### 11.3 Micro Near

### 읽어줄 문구

> 이 condition에서는 머리 근처에서 micro control을 사용합니다.
>
> 화면에 나온 것처럼 손을 머리 근처에 두어 주세요.
>
> Micro control에서는 팔을 최대한 고정하고, phone 입력으로 미세하게 조정해 주세요.

### 11.4 Micro Side

### 읽어줄 문구

> 이 condition에서는 몸 옆에서 micro control을 사용합니다.
>
> 화면에 나온 것처럼 손을 몸 옆에 두어 주세요.
>
> Micro control에서는 팔을 최대한 고정하고, phone 입력으로 미세하게 조정해 주세요.

## 12. Task별 설명 문구

본 실험 전에 이 설명을 먼저 읽어주고, 필요할 때만 다시 짧게 상기시킵니다.

### 12.1 Placement

### 읽어줄 문구

> Placement에서는 tool을 target 위치에 최대한 가깝게 맞춰 주세요.
>
> 그 다음 `triple tap`으로 현재 시도를 제출해 주세요.

### Micro placement 추가 설명

### 읽어줄 문구

> Micro placement에서는 한 번 tap 하면 grab, 다시 tap 하면 release입니다.
>
> 좌우 swipe는 좌우 이동입니다.
>
> 위아래 swipe는 tool을 이동시키는 조작입니다.
>
> `Double tap`으로 위아래 swipe 조작을 위아래 이동과 앞뒤 이동 사이에서 전환할 수 있습니다.

### 12.2 Rotation

### 읽어줄 문구

> Rotation에서는 tool의 회전을 target과 최대한 비슷하게 맞춰 주세요.
>
> Rotation에서는 tool의 위치를 target 위치에 정확히 맞출 필요는 없고, 각도만 조정하면 됩니다.
>
> 그 다음 `triple tap`으로 현재 시도를 제출해 주세요.

### Micro rotation 추가 설명

### 읽어줄 문구

> Micro rotation에서는 좌우 swipe로 yaw를 조정합니다.
>
> 위아래 swipe로 roll을 조정합니다.
>
> `Double tap`으로 위아래 swipe 조작을 roll에서 pitch로 전환할 수 있습니다.

### 12.3 Scaling

### 읽어줄 문구

> Scaling에서는 tool의 크기를 target과 최대한 비슷하게 맞춰 주세요.
>
> 그 다음 `triple tap`으로 현재 시도를 제출해 주세요.

### Micro scaling 추가 설명

### 읽어줄 문구

> Micro scaling에서는 위로 swipe 하면 tool이 작아지고, 아래로 swipe 하면 tool이 커집니다.

## 13. Practice 안내

### 진행자

- 참가자가 화면에 나온 practice instruction을 따르도록 합니다
- 참가자가 혼란스러워하지 않는다면 과도하게 설명하지 않습니다
- practice는 control에 익숙해지는 단계라고만 안내합니다

### 읽어줄 문구

> Task phase 전에 짧은 practice round가 나올 수 있습니다.
>
> Practice 동안에는 control에 익숙해지는 데 집중해 주세요.
>
> 현재 practice round가 끝났다고 느껴지면 `triple tap` 해 주세요.

### Practice 종료 후

### 읽어줄 문구

> 이제 main task가 시작됩니다.

## 14. Main trial 안내

### 읽어줄 문구

> Target을 최대한 정확하게 맞춰 주세요.
>
> 현재 시도를 제출할 준비가 되면 `triple tap` 해 주세요.

### 참가자가 너무 빨리 제출할 때

### 읽어줄 문구

> `Triple tap` 하기 전에 최대한 더 정확하게 맞춰 주세요.

### 참가자가 목표를 애매하게 이해할 때

### 읽어줄 문구

> 단순히 어느 정도 가까워지는 것이 목표는 아닙니다.
>
> 가능한 한 정확하게 맞춘 뒤, 제출할 준비가 되면 `triple tap` 해 주세요.

## 15. Condition 후 설문

### 진행자

- 각 condition이 끝날 때마다 questionnaire를 제공합니다
- condition마다 같은 방식으로 안내합니다
- 방금 끝난 condition을 이후 condition과 비교해서 설명하지 않습니다

### 읽어줄 문구

> 지금 condition이 끝났습니다.
>
> 잠깐 쉬시고, 방금 사용한 condition에 대한 짧은 설문을 작성해 주세요.

### 필요하면

### 읽어줄 문구

> 방금 끝난 condition만 기준으로 답해 주세요.

## 16. 최종 설문

### 진행자

- 네 개 condition이 모두 끝난 뒤 최종 설문을 제공합니다
- 전체 선호도, 편안함, workload, confidence, 자유 의견 등을 받습니다

### 읽어줄 문구

> 이제 모든 study condition이 끝났습니다.
>
> 전체 경험에 대한 최종 설문을 작성해 주세요.
>
> 쉬웠던 점, 어려웠던 점, 편했던 점, 헷갈렸던 점이 있으면 자유롭게 적어 주셔도 됩니다.

## 17. 세션 종료

### 진행자

- phone과 실험 자료를 회수합니다
- 모든 문서가 완료되었는지 확인합니다
- 데이터가 저장되었는지 확인합니다
- 필요하면 보상을 진행합니다
- 참가자에게 감사 인사를 전합니다

### 읽어줄 문구

> 참여해 주셔서 다시 한 번 감사합니다.

## 18. 문제 발생 시 사용할 문구

필요할 때만 사용합니다.

### QR 스캔이 되지 않을 때

> Workspace alignment를 다시 맞추는 동안 잠시만 기다려 주세요.

### 참가자가 condition 시작 절차를 잊었을 때

> 화면에 나온 자세를 먼저 맞추고, 그 다음 `triple tap` 해서 condition을 시작해 주세요.

### Condition 시작 직후 참가자가 많이 움직일 때

> 시작하려고 `triple tap` 한 뒤에는 시스템이 준비될 때까지 잠깐만 가만히 있어 주세요.

### 참가자가 task 중 언제 `triple tap` 해야 하는지 묻는 경우

> 지금 시도가 target에 최대한 가깝다고 느껴질 때 `triple tap` 하시면 됩니다.

### 참가자가 practice도 기록되는지 묻는 경우

> 아닙니다. Practice는 control에 익숙해지기 위한 단계이고, 그 다음에 main task가 진행됩니다.

## 19. 진행자 빠른 참고

- 실험 유형: `within-subject`
- Condition 시작: 자세 이미지에 맞춘 뒤 `triple tap to start`
- Condition 시작 직후: 잠깐 정지 자세 유지
- Main submit 동작: `triple tap`
- 현재 workflow: 각 condition에서 `Placement -> Rotation -> Scaling`, 각 phase당 `5 tools`
- Latin square sequence를 `Participant ID`에 연결할 경우 ID에 숫자가 포함되어야 함
