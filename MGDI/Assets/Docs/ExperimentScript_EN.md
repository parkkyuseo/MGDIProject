# Experiment Script

## 1. Purpose

This document is a facilitator guide with participant-facing spoken lines.

It is written in simple, easy-to-read English so the facilitator can read most lines directly to the participant.

This script assumes:

- a within-subject study
- 4 conditions per participant
- condition order is counterbalanced with a Latin square
- the base order for counterbalancing is `Macro Near -> Macro Side -> Micro Near -> Micro Side`
- each condition includes `Placement`, `Rotation`, and `Scaling`
- each phase currently uses `2 tools` in the workflow

Important note:

- If the Latin square order is tied to `Participant ID`, the ID should include a number such as `P01`, `P12`, or `P24`.
- In the current code, the numeric part of the `Participant ID` is what can be used to determine the sequence.

### Condition order lookup by Participant ID

This lookup assumes the base order is:

- `A = Macro Near`
- `B = Macro Side`
- `C = Micro Near`
- `D = Micro Side`

The code uses the numeric part of the participant ID.

Examples:

- `P01` uses `1`
- `P15` uses `15`
- `P24` uses `24`

For 4 conditions, the sequence repeats every 4 participant numbers.

| Participant number | Sequence | Condition order |
| --- | --- | --- |
| `1, 5, 9, 13, 17, ...` | `A -> D -> B -> C` | `Macro Near -> Micro Side -> Macro Side -> Micro Near` |
| `2, 6, 10, 14, 18, ...` | `B -> A -> C -> D` | `Macro Side -> Macro Near -> Micro Near -> Micro Side` |
| `3, 7, 11, 15, 19, ...` | `C -> B -> D -> A` | `Micro Near -> Macro Side -> Micro Side -> Macro Near` |
| `4, 8, 12, 16, 20, ...` | `D -> C -> A -> B` | `Micro Side -> Micro Near -> Macro Near -> Macro Side` |

Quick example:

- `P15` uses participant number `15`
- `15` falls in the `3, 7, 11, 15, 19, ...` group
- so `P15` uses `C -> B -> D -> A`
- that means `Micro Near -> Macro Side -> Micro Side -> Macro Near`

### Participant ID quick lookup (`P01-P50`)

Use this table when you want to know the condition order immediately without doing the modulo calculation by hand.

| Participant ID | Sequence | Condition order |
| --- | --- | --- |
| `P01` | `A -> D -> B -> C` | `Macro Near -> Micro Side -> Macro Side -> Micro Near` |
| `P02` | `B -> A -> C -> D` | `Macro Side -> Macro Near -> Micro Near -> Micro Side` |
| `P03` | `C -> B -> D -> A` | `Micro Near -> Macro Side -> Micro Side -> Macro Near` |
| `P04` | `D -> C -> A -> B` | `Micro Side -> Micro Near -> Macro Near -> Macro Side` |
| `P05` | `A -> D -> B -> C` | `Macro Near -> Micro Side -> Macro Side -> Micro Near` |
| `P06` | `B -> A -> C -> D` | `Macro Side -> Macro Near -> Micro Near -> Micro Side` |
| `P07` | `C -> B -> D -> A` | `Micro Near -> Macro Side -> Micro Side -> Macro Near` |
| `P08` | `D -> C -> A -> B` | `Micro Side -> Micro Near -> Macro Near -> Macro Side` |
| `P09` | `A -> D -> B -> C` | `Macro Near -> Micro Side -> Macro Side -> Micro Near` |
| `P10` | `B -> A -> C -> D` | `Macro Side -> Macro Near -> Micro Near -> Micro Side` |
| `P11` | `C -> B -> D -> A` | `Micro Near -> Macro Side -> Micro Side -> Macro Near` |
| `P12` | `D -> C -> A -> B` | `Micro Side -> Micro Near -> Macro Near -> Macro Side` |
| `P13` | `A -> D -> B -> C` | `Macro Near -> Micro Side -> Macro Side -> Micro Near` |
| `P14` | `B -> A -> C -> D` | `Macro Side -> Macro Near -> Micro Near -> Micro Side` |
| `P15` | `C -> B -> D -> A` | `Micro Near -> Macro Side -> Micro Side -> Macro Near` |
| `P16` | `D -> C -> A -> B` | `Micro Side -> Micro Near -> Macro Near -> Macro Side` |
| `P17` | `A -> D -> B -> C` | `Macro Near -> Micro Side -> Macro Side -> Micro Near` |
| `P18` | `B -> A -> C -> D` | `Macro Side -> Macro Near -> Micro Near -> Micro Side` |
| `P19` | `C -> B -> D -> A` | `Micro Near -> Macro Side -> Micro Side -> Macro Near` |
| `P20` | `D -> C -> A -> B` | `Micro Side -> Micro Near -> Macro Near -> Macro Side` |
| `P21` | `A -> D -> B -> C` | `Macro Near -> Micro Side -> Macro Side -> Micro Near` |
| `P22` | `B -> A -> C -> D` | `Macro Side -> Macro Near -> Micro Near -> Micro Side` |
| `P23` | `C -> B -> D -> A` | `Micro Near -> Macro Side -> Micro Side -> Macro Near` |
| `P24` | `D -> C -> A -> B` | `Micro Side -> Micro Near -> Macro Near -> Macro Side` |
| `P25` | `A -> D -> B -> C` | `Macro Near -> Micro Side -> Macro Side -> Micro Near` |
| `P26` | `B -> A -> C -> D` | `Macro Side -> Macro Near -> Micro Near -> Micro Side` |
| `P27` | `C -> B -> D -> A` | `Micro Near -> Macro Side -> Micro Side -> Macro Near` |
| `P28` | `D -> C -> A -> B` | `Micro Side -> Micro Near -> Macro Near -> Macro Side` |
| `P29` | `A -> D -> B -> C` | `Macro Near -> Micro Side -> Macro Side -> Micro Near` |
| `P30` | `B -> A -> C -> D` | `Macro Side -> Macro Near -> Micro Near -> Micro Side` |
| `P31` | `C -> B -> D -> A` | `Micro Near -> Macro Side -> Micro Side -> Macro Near` |
| `P32` | `D -> C -> A -> B` | `Micro Side -> Micro Near -> Macro Near -> Macro Side` |
| `P33` | `A -> D -> B -> C` | `Macro Near -> Micro Side -> Macro Side -> Micro Near` |
| `P34` | `B -> A -> C -> D` | `Macro Side -> Macro Near -> Micro Near -> Micro Side` |
| `P35` | `C -> B -> D -> A` | `Micro Near -> Macro Side -> Micro Side -> Macro Near` |
| `P36` | `D -> C -> A -> B` | `Micro Side -> Micro Near -> Macro Near -> Macro Side` |
| `P37` | `A -> D -> B -> C` | `Macro Near -> Micro Side -> Macro Side -> Micro Near` |
| `P38` | `B -> A -> C -> D` | `Macro Side -> Macro Near -> Micro Near -> Micro Side` |
| `P39` | `C -> B -> D -> A` | `Micro Near -> Macro Side -> Micro Side -> Macro Near` |
| `P40` | `D -> C -> A -> B` | `Micro Side -> Micro Near -> Macro Near -> Macro Side` |
| `P41` | `A -> D -> B -> C` | `Macro Near -> Micro Side -> Macro Side -> Micro Near` |
| `P42` | `B -> A -> C -> D` | `Macro Side -> Macro Near -> Micro Near -> Micro Side` |
| `P43` | `C -> B -> D -> A` | `Micro Near -> Macro Side -> Micro Side -> Macro Near` |
| `P44` | `D -> C -> A -> B` | `Micro Side -> Micro Near -> Macro Near -> Macro Side` |
| `P45` | `A -> D -> B -> C` | `Macro Near -> Micro Side -> Macro Side -> Micro Near` |
| `P46` | `B -> A -> C -> D` | `Macro Side -> Macro Near -> Micro Near -> Micro Side` |
| `P47` | `C -> B -> D -> A` | `Micro Near -> Macro Side -> Micro Side -> Macro Near` |
| `P48` | `D -> C -> A -> B` | `Micro Side -> Micro Near -> Macro Near -> Macro Side` |
| `P49` | `A -> D -> B -> C` | `Macro Near -> Micro Side -> Macro Side -> Micro Near` |
| `P50` | `B -> A -> C -> D` | `Macro Side -> Macro Near -> Micro Near -> Micro Side` |

## 2. Facilitator Summary

Each participant completes all four conditions.

For each condition:

- the participant matches the required posture
- the participant triple taps to begin the condition
- the system may show a short practice before a task phase
- the participant completes `Placement`, `Rotation`, and `Scaling`
- each phase is completed for `2 tools` in the current workflow
- the participant then completes a short between-condition questionnaire

Core participant rules:

- `Triple tap` submits the current attempt.
- In `Macro`, the participant mainly uses larger arm and hand movements.
- In `Micro`, the participant should keep the arm as still as possible and use phone swipes.
- After the participant triple taps to begin a condition, the system briefly checks phone connection and stillness before the task starts.

## 3. Before The Session

### Facilitator checklist

- Prepare the headset, phone, and study computer.
- Prepare the consent form.
- Prepare the demographic questionnaire.
- Prepare the between-condition questionnaire.
- Prepare the final questionnaire.
- Prepare the debrief sheet.
- Prepare the participant compensation form, if needed.
- Confirm the participant's counterbalanced condition order.
- Assign a `Participant ID` with a numeric part.
- Confirm that the phone app is available and the QR workspace setup is ready.
- Check the basket and tool locations in the workspace.
- Confirm that the session starts from the participant ID screen.

### Suggested system check

Before the participant arrives, confirm:

- the start screen asks for `Participant ID`
- the runtime begins after QR scanning
- the phone connects correctly
- the posture example images are visible
- the phone can detect `triple tap`

## 4. Welcome And Consent

### Facilitator

- Welcome the participant.
- Confirm their name and appointment.
- Ask whether they are comfortable continuing.
- Give the consent form.

### Say

> Hello, thanks for coming in today.
>
> In this study, you will use a mixed reality system and a phone-based input method to complete object manipulation tasks.
>
> Before we begin, please read this consent form carefully.
>
> Feel free to ask questions at any time.
>
> If you are happy to participate, please sign the form.

### After consent

### Say

> Thank you.
>
> Next, I will ask you to complete a short background questionnaire.

## 5. Demographic Questionnaire

### Facilitator

- Give the demographic questionnaire.
- Wait until it is completed.
- Answer procedural questions, but do not explain the study hypotheses.

### Say

> Please complete this short background questionnaire.
>
> Just let me know when you are done.

## 6. Study Overview

### Say

> In this study, you will complete the same types of tasks in several different conditions.
>
> The conditions change how you control the system and where you keep your hand.
>
> In every condition, your goal is to match the target as closely as possible.
>
> When you are ready, you will submit your current attempt with a triple tap.
>
> This is a within-subject study, so you will try all of the conditions.
>
> The order is counterbalanced across participants.

### Short version

> You will do the same tasks in several conditions.
>
> Please try to be as accurate as you can in every condition.
>
> When you are ready to submit your current attempt, triple tap.

## 7. Device Setup

### Facilitator

- Hand the phone to the participant.
- Confirm that the participant can hold it comfortably.
- Explain how to launch the phone app, if needed.
- Enter or confirm the assigned `Participant ID`.

### Say

> I will now give you the phone we use in the study.
>
> Please hold it in a comfortable way.
>
> We will start by entering your participant ID.

### Facilitator note

On the start screen:

- the participant enters the assigned `Participant ID`
- the screen then continues to the runtime scene
- the runtime scene shows `Scan the QR code to begin.`

## 8. Start Of Runtime

### Facilitator

- Ask the participant to scan the QR code when the runtime instruction appears.
- Confirm that the workspace is aligned correctly.

### Say

> Please scan the QR code to begin.
>
> Once the workspace is ready, we will start the study conditions.

## 9. General Task Rules

### Say

> In every task, please try to match the target as closely as possible before you submit.
>
> There is no separate start button for each task. Once the task begins, you can start right away.
>
> When you are ready to submit your current attempt, please triple tap.
>
> If a short practice appears, just follow the instruction on the screen. The practice is only for familiarization.

### Say

> Between conditions, you will have a short break and a short questionnaire.
>
> If anything feels wrong, confusing, or uncomfortable, please tell me right away.

## 10. Condition Introduction

### Facilitator

- Before each condition, remind the participant that the posture image on the screen is the reference.
- Do not rush the participant during posture matching.
- After the participant triple taps to begin the condition, wait for the system to complete the phone connection and stillness check.

### Say

> At the start of each condition, please match the posture shown in the image.
>
> When your posture matches the instruction, triple tap to start.
>
> After that, please hold still for a moment until the system is ready.

## 11. Condition Explanation Script

Use the relevant explanation before the participant starts each condition.

### 11.1 Macro Near

### Say

> In this condition, you will use macro control near your head.
>
> Please keep your hand near your head, as shown in the image.
>
> In macro control, you will mainly use larger arm and hand movements.

### 11.2 Macro Side

### Say

> In this condition, you will use macro control at the side of your body.
>
> Please keep your hand at the side of your body, as shown in the image.
>
> In macro control, you will mainly use larger arm and hand movements.
>
> In this side condition, the movement mapping changes.
>
> Up and down movement is remapped with forward and backward movement.

### 11.3 Micro Near

### Say

> In this condition, you will use micro control near your head.
>
> Please keep your hand near your head, as shown in the image.
>
> In micro control, try to keep your arm as still as possible and use phone input for fine adjustment.

### 11.4 Micro Side

### Say

> In this condition, you will use micro control at the side of your body.
>
> Please keep your hand at the side of your body, as shown in the image.
>
> In micro control, try to keep your arm as still as possible and use phone input for fine adjustment.

## 12. Task Explanation Script

Use these explanations before the participant begins the main experiment, and repeat them only if needed.

### 12.1 Placement

### Say

> In placement, move the tool so it is as close to the target position as possible.
>
> Then triple tap to submit your current attempt.

### Micro placement add-on

### Say

> In micro placement, tap once to grab and tap again to release.
>
> Swipe left and right to move left and right.
>
> Swipe up and down to move the tool.
>
> You can double tap to switch the up-and-down swipe control between up-and-down movement and front-and-back movement.

### 12.2 Rotation

### Say

> In rotation, match the tool to the target rotation as closely as possible.
>
> Then triple tap to submit your current attempt.

### Micro rotation add-on

### Say

> In micro rotation, swipe left and right to adjust yaw.
>
> Swipe up and down to adjust roll.
>
> You can double tap to switch the up-and-down swipe control from roll to pitch.

### 12.3 Scaling

### Say

> In scaling, match the tool to the target size as closely as possible.
>
> Then triple tap to submit your current attempt.

### Micro scaling add-on

### Say

> In micro scaling, swipe up to make the tool smaller.
>
> Swipe down to make the tool bigger.

## 13. Practice Script

### Facilitator

- Let the participant follow the on-screen practice instruction.
- Do not over-explain unless the participant is confused.
- Remind the participant that practice is only for getting familiar with the control.

### Say

> You may see a short practice round before a task phase.
>
> During practice, just get familiar with the control.
>
> When you are done with the current practice round, triple tap.

### When practice ends

### Say

> The main task starts now.

## 14. Main Trial Script

### Say

> Please do your best to match the target as closely as possible.
>
> When you are ready to submit your current attempt, triple tap.

### If the participant submits too early

### Say

> Please try to get as close as you can before you triple tap.

### If the participant seems unsure about the goal

### Say

> The goal is not just to get close enough.
>
> Please try to match the target as accurately as you can, then triple tap when you are ready to submit.

## 15. Between-Condition Questionnaire

### Facilitator

- At the end of each condition, give the participant the between-condition questionnaire.
- Keep the wording consistent across conditions.
- Do not compare the current condition to later conditions out loud.

### Say

> That condition is now finished.
>
> Please take a short break.
>
> While you rest, please complete this short questionnaire about the condition you just used.

### If needed

### Say

> Please answer based only on the condition you just completed.

## 16. Final Questionnaire

### Facilitator

- After all four conditions are complete, give the final questionnaire.
- Ask for overall preference, comfort, workload, confidence, and any open comments.

### Say

> You have now completed all of the study conditions.
>
> Please complete this final questionnaire about your overall experience.
>
> You can also share any comments about what felt easy, difficult, comfortable, or confusing.

## 17. Debrief

### Say

> Thank you for completing the study.
>
> This study compares different input conditions for object manipulation tasks in mixed reality.
>
> We are interested in how the different conditions affect performance and user experience.
>
> If you have any final comments or questions, I would be happy to hear them now.

## 18. End Of Session

### Facilitator

- Collect the phone and any study materials.
- Confirm that all forms are complete.
- Confirm that the data was saved.
- Provide compensation, if applicable.
- Thank the participant.

### Say

> Thank you again for your time and participation.

## 19. Troubleshooting Lines

Use these only when needed.

### If QR scanning does not work

> Please give me one moment while I reset the workspace alignment.

### If the participant forgets the condition start step

> Please match the posture shown in the image, then triple tap to start the condition.

### If the participant moves too much right after the start triple tap

> After you triple tap to start, please hold still for a moment until the system is ready.

### If the participant asks when to triple tap during a task

> Triple tap when you feel your current attempt is as close to the target as you can make it.

### If the participant asks whether the practice counts

> No. The practice is only for familiarization. The main task comes after the practice.

## 20. Facilitator Quick Reference

- Study type: `within-subject`
- Base order for counterbalancing: `Macro Near -> Macro Side -> Micro Near -> Micro Side`
- Condition start: match posture image, then `triple tap to start`
- After condition start: participant should hold still briefly
- Main submit action: `triple tap`
- Current workflow per condition: `Placement -> Rotation -> Scaling`, with `2 tools` per phase
- `Participant ID` should include a number if the Latin square sequence is linked to the participant ID
