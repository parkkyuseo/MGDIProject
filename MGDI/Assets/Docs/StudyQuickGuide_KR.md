# 스터디 빠른 안내 (KR)

## 1. 개요

- 이 스터디는 4개 컨디션에서 3개 태스크를 수행합니다.
- 컨디션: `Macro - Near Head`, `Macro - Side Of Body`, `Micro - Near Head`, `Micro - Side Of Body`
- 태스크: `Placement`, `Rotation`, `Scaling`
- 컨디션 순서는 `Participant ID`에 따라 달라질 수 있습니다.
- `Macro`와 `Micro` 모두 모든 태스크에서 제출은 `triple tap`입니다.

## 2. 진행 절차

1. 시작 화면에서 배정받은 `Participant ID`를 입력합니다.
2. 필요하면 `Edit` 버튼으로 수정하고, `Continue`를 눌러 시작합니다.
3. 런타임 화면에서 `Scan the QR code to begin.` 안내가 나오면 HoloLens workspace QR 코드를 스캔합니다.
4. 각 컨디션 시작 시 현재 컨디션 이름이 표시됩니다. 설명을 들은 뒤 phone에서 `triple tap` 해서 다음 단계로 넘어갑니다.
5. Phone app을 실행 또는 재실행하고, phone으로 QR 코드를 스캔한 뒤 HoloLens의 `Phone QR detected.` 안내를 기다립니다.
6. 자세 안내와 예시 이미지에 맞춰 자세를 잡은 뒤 `triple tap` 해서 시작합니다. HoloLens에서 `Triple tap detected. Hold still.` 안내가 나오면 잠시 가만히 있습니다.
7. 각 태스크 시작 전 현재 태스크 이름이 표시됩니다. 설명을 들은 뒤 phone에서 `triple tap` 해서 다음 단계로 넘어갑니다.
8. 각 태스크 시작 전 짧은 `practice`가 나올 수 있습니다. `practice`는 기록되지 않습니다.
9. 각 main task의 첫 main trial 전에 HoloLens가 `Main Placement Task Starts Now`, `Main Rotation Task Starts Now`, 또는 `Main Scaling Task Starts Now`를 안내합니다.
10. 목표를 맞춘 뒤 `triple tap`으로 제출합니다.
11. 한 컨디션이 끝나면 break/questionnaire 단계를 마친 뒤 `triple tap` 해서 다음 컨디션으로 넘어갑니다.

## 3. 컨디션 개요

### Macro - Near Head

- 팔과 손의 큰 움직임으로 조작합니다.
- 손 위치는 머리 가까이 유지합니다.

#### Spoken script (EN)

> In this condition, keep your hand near your head.
>
> You will use larger phone and arm movements to control the proxy hand.
>
> After this explanation, triple tap to continue.

![Macro Near Head](../Image/MacroNearHead.png)

### Macro - Side Of Body

- 팔과 손의 큰 움직임으로 조작합니다.
- 손 위치는 몸 옆(side of body)에 둡니다.
- 이 컨디션에서는 입력이 리맵됩니다.
- Phone의 좌우 움직임은 proxy hand의 좌우 움직임으로 유지됩니다.
- Phone의 위아래 움직임은 proxy hand의 앞뒤 움직임을 조정합니다.
- Phone의 앞뒤 움직임은 proxy hand의 위아래 움직임을 조정합니다.

#### Spoken script (EN)

> In this condition, keep your hand at the side of your body.
>
> You will use larger phone and arm movements.
>
> The movement is changed here: left and right stay left and right, up and down move forward and back, and forward and back move up and down.
>
> After this explanation, triple tap to continue.

![Macro Side Of Body](../Image/MacroSideBody.png)

### Micro - Near Head

- 손 위치는 머리 가까이 유지합니다.
- 세밀한 조작은 휴대폰 화면을 스와이프해서 수행합니다.

#### Spoken script (EN)

> In this condition, keep your hand near your head.
>
> Try to keep your arm still, and use small phone swipes for fine control.
>
> After this explanation, triple tap to continue.

![Micro Near Head](../Image/MacroNearHead.png)

### Micro - Side Of Body

- 손 위치는 몸 옆(side of body)에 둡니다.
- 세밀한 조작은 휴대폰 화면을 스와이프해서 수행합니다.

#### Spoken script (EN)

> In this condition, keep your hand at the side of your body.
>
> Try to keep your arm still, and use small phone swipes for fine control.
>
> After this explanation, triple tap to continue.

![Micro Side Of Body](../Image/MicroSideBody.png)

## 4. 태스크 개요

### Placement

- 도구를 목표 위치에 맞춥니다.
- `Micro`에서는 휴대폰 화면을 좌우/상하로 스와이프해서 위치를 조정합니다.
- 필요하면 `double tap`으로 이동 평면을 바꿔 깊이(앞/뒤) 방향도 조정할 수 있습니다.
- 목표 위치를 맞춘 뒤 `triple tap`으로 제출합니다.

#### Spoken script (EN)

> In this task, move the tool to the target position.
>
> Try to place it as close to the target as possible.
>
> When you are done, triple tap to submit.

Micro add-on:

> In micro placement, tap once to grab and tap again to release.
>
> Swipe to move the tool. Double tap if you need to change the movement plane.

### Rotation

- 도구의 방향을 목표 방향에 맞춥니다.
- `Micro`에서는 좌우 스와이프로 좌우 회전(yaw), 상하 스와이프로 기울기 회전(roll)을 조정합니다.
- 필요하면 `double tap`으로 상하 스와이프 축을 `roll`과 `pitch` 사이에서 전환할 수 있습니다.
- 목표 방향을 맞춘 뒤 `triple tap`으로 제출합니다.

#### Spoken script (EN)

> In this task, match the tool's angle to the target.
>
> You do not need to match the position exactly. Focus on the rotation.
>
> When you are done, triple tap to submit.

Micro add-on:

> In micro rotation, swipe left and right to turn the tool.
>
> Swipe up and down for another rotation direction. Double tap to switch that direction.

### Scaling

- 도구의 크기를 목표 크기에 맞춥니다.
- `Micro`에서는 상하 스와이프로 크기를 키우거나 줄입니다.
- 목표 크기를 맞춘 뒤 `triple tap`으로 제출합니다.

#### Spoken script (EN)

> In this task, match the tool's size to the target size.
>
> Make the tool as close to the target size as possible.
>
> When you are done, triple tap to submit.

Micro add-on:

> In micro scaling, swipe up to make the tool smaller, and swipe down to make it bigger.

## 5. 참고

- `Macro`에서는 팔과 손의 움직임으로 조작합니다.
- `Micro`에서는 휴대폰 스와이프로 세밀하게 조작합니다.
- 프록시 핸드가 오브젝트에 충분히 가까우면 잡을 수 있습니다.
- `Macro`에서는 화면을 터치하고 있는 동안 오브젝트를 잡고, 터치를 놓으면 오브젝트를 놓습니다.
- `Micro`에서는 한 번 탭해서 오브젝트를 잡고, 다시 한 번 탭해서 오브젝트를 놓습니다.
- 새 컨디션에서 phone app을 다시 켠 경우, 자세를 맞추기 전에 phone으로 QR 코드를 다시 스캔합니다.
- 자세 안내 이미지가 표시되는 동안과 HoloLens가 잠시 가만히 있으라고 안내하는 동안에는 자세를 크게 바꾸지 않는 것이 좋습니다.
- QR 인식이 잘 안 되거나 `Phone QR detected.` 안내가 나오지 않거나 화면 또는 입력이 이상하면 진행자에게 바로 알리면 됩니다.

## 6. 최고 중요정보

- 시작 전에 `Participant ID`를 입력하고 HoloLens workspace `QR`을 스캔합니다.
- 각 컨디션마다 condition 설명을 듣고 `triple tap` 해서 넘어간 뒤, phone으로 QR 코드를 스캔하고 `Phone QR detected.` 안내를 확인한 다음 자세를 맞추고 `triple tap` 합니다.
- 현재 컨디션에 맞게 손 위치를 유지합니다: `Near Head` 또는 `Side Of Body`
- HoloLens에서 `Triple tap detected. Hold still.` 안내가 나오면 잠시 가만히 있습니다.
- `Macro`는 화면을 누르고 있는 동안 잡고, `Micro`는 탭으로 잡고 놓습니다.
- `Micro`는 휴대폰 스와이프로 조작합니다.
- `Macro - Side Of Body`에서는 phone 좌우 움직임은 좌우로 유지되고, phone 위아래 움직임은 앞뒤를, phone 앞뒤 움직임은 위아래를 조정합니다.
- 모든 태스크의 제출은 `triple tap`입니다.
