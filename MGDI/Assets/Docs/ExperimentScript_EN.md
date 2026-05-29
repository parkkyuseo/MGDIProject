# Experiment Script

## 1. Setup

- a within-subject study
- 4 conditions per participant
- condition order is counterbalanced with a Latin square
- the base order for counterbalancing is `Macro Near -> Macro Side -> Micro Near -> Micro Side`
- each condition includes `Placement`, `Rotation`, and `Scaling`
- each phase currently uses `5 tools` in the workflow
- If the Latin square order is tied to `Participant ID`, the ID should include a number such as `P01`, `P12`, or `P24`.
- Use the numeric part of the `Participant ID` to determine the sequence.

## 2. Facilitator Summary

Each participant completes all four conditions.

For each condition:

- the system shows the condition name, and the condition is explained before the participant continues
- the participant triple taps to continue from the condition explanation
- the participant opens or restarts the phone app, scans the QR code with the phone, and waits for the HoloLens `Phone QR detected.` announcement
- the participant matches the required posture
- the participant triple taps to begin the condition
- the system says `Triple tap detected. Hold still.` and briefly checks phone connection and stillness
- before each task phase, the system shows the task name, and the task is explained before the participant continues
- the participant triple taps to continue from the task explanation
- the system may show a short practice before a task phase
- the participant completes `Placement`, `Rotation`, and `Scaling`
- each phase is completed for `5 tools` in the current workflow
- the participant then completes a short questionnaire

Core participant rules:

- During task trials, `triple tap` submits the current attempt.
- On condition, task, and break screens, `triple tap` continues to the next step.
- In `Macro`, the participant mainly uses larger arm and hand movements.
- In `Micro`, the participant should keep the arm as still as possible and use phone swipes.
- If the phone app is restarted between conditions, the participant must scan the QR code again with the phone before starting that condition.
- After the participant triple taps to begin a condition, the system briefly checks phone connection and stillness before the task sequence starts.

## 3. Before The Session

### Facilitator checklist

- Prepare the headset, phone, and study computer.
- Prepare the consent form.
- Prepare the questionnaires. (including one questionnaire for each condition)
- Prepare the participant compensation form, if needed.
- Confirm the participant's counterbalanced condition order.
- Assign a `Participant ID` with a numeric part.
- Confirm that the phone app is available, can detect the QR code, and the QR workspace setup is ready.
- Check the basket and tool locations in the workspace.
- Confirm that the session starts from the participant ID screen.

### Suggested system check

Before the participant arrives, confirm:

- the start screen asks for `Participant ID`
- the runtime begins after QR scanning
- the phone connects correctly
- the HoloLens announces `Phone QR detected.` after the phone scans the QR code
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
> When your current attempt is as close to the target as you can make it, submit it with a triple tap.
>
> This is a within-subject study, so you will try all of the conditions.
>
> The order is counterbalanced across participants.

### Short version

> You will do the same tasks in several conditions.
>
> Please try to be as accurate as you can in every condition.
>
> When your current attempt is as accurate as you can make it, triple tap to submit.

## 7. Device Setup

### Facilitator

- Open the app and enter the assigned `Participant ID`.
- Hand the HoloLens to the participant.
- After that, hand the phone to the participant.
- Confirm that the participant can hold the phone comfortably.

### Say

> I will first open the app and enter your participant ID.
>
> Then I will hand you the HoloLens.
>
> After that, I will give you the phone.
>
> Please hold the phone in a comfortable way.

### Facilitator note

On the start screen:

- I enter the assigned `Participant ID`
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
> When your current attempt is as close to the target as you can make it, please triple tap.
>
> If a short practice appears, just follow the instruction on the screen. The practice is only for familiarization.

### Say

> Between conditions, you will have a short break and a short questionnaire.
>
> If anything feels wrong, confusing, or uncomfortable, please tell me right away.

## 10. Condition Introduction

### Facilitator

- At the start of each condition, wait on the condition title screen while the condition is explained.
- Have the participant triple tap on the phone to continue from the condition explanation.
- Then have the participant open or restart the phone app if needed.
- Ask the participant to scan the QR code with the phone and wait for the HoloLens `Phone QR detected.` announcement.
- Remind the participant that the posture image on the screen is the reference.
- Do not rush the participant during posture matching.
- After the participant triple taps to begin the condition, the HoloLens should say `Triple tap detected. Hold still.`
- Wait for the system to complete the phone connection and stillness check.

### Say

> The next screen will show the current condition.
>
> I will explain the condition first.
>
> After the explanation, please triple tap on the phone to continue.
>
> Then please open the phone app and scan the QR code with the phone.
>
> Please wait until the HoloLens says, "Phone QR detected."
>
> After that, please match the posture shown in the image.
>
> When your posture matches the instruction, triple tap to start.
>
> After the HoloLens says "Triple tap detected. Hold still," please hold still for a moment.

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
> Phone left and right movement stays mapped to proxy-hand left and right movement.
>
> Phone up and down movement controls proxy-hand forward and backward movement.
>
> Phone forward and backward movement controls proxy-hand up and down movement.

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

Use these explanations when the task title screen appears. After the explanation, the participant triple taps on the phone to continue.

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
> In rotation, you do not need to match the tool to the target position. You only need to adjust the angle.
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
> The practice screen will name the task and then show what to match before asking for a triple tap.
>
> During practice, just get familiar with the control.
>
> When you are done with the current practice round, triple tap.

### Facilitator note

Practice can appear before `Placement`, `Rotation`, or `Scaling`. In the second Macro condition, `Rotation` and `Scaling` can also include practice even if the first Macro condition already included practice.

### When practice ends

### Say

> The main task starts now.

## 14. Main Trial Script

### Say

> The HoloLens will announce the main task once before the first main trial, for example, "Main Placement Task Starts Now."
>
> Please do your best to match the target as closely as possible.
>
> When your current attempt is as close to the target as you can make it, triple tap.

### If the participant submits too early

### Say

> Please try to get as close as you can before you triple tap.

### If the participant seems unsure about the goal

### Say

> The goal is not just to get close enough.
>
> Please try to match the target as accurately as you can, then triple tap to submit.

## 15. Questionnaire After Each Condition

### Facilitator

- At the end of each condition, give the participant the questionnaire.
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

## 17. End Of Session

### Facilitator

- Collect the phone and any study materials.
- Confirm that all forms are complete.
- Confirm that the data was saved.
- Provide compensation, if applicable.
- Thank the participant.

### Say

> Thank you again for your time and participation.

## 18. Troubleshooting Lines

Use these only when needed.

### If QR scanning does not work

> Please give me one moment while I reset the workspace alignment.

### If the phone QR detection announcement does not happen

> Please keep the phone app open and point the phone camera at the QR code until the HoloLens says, "Phone QR detected."

### If the participant forgets the condition start step

> First listen to the condition explanation, then triple tap to continue. After that, scan the QR code with the phone, wait for "Phone QR detected," match the posture shown in the image, then triple tap to start the condition.

### If the participant moves too much right after the start triple tap

> After you triple tap to start, the HoloLens should say "Triple tap detected. Hold still." Please hold still for a moment after that.

### If the participant asks when to triple tap during a task

> Triple tap when you feel your current attempt is as close to the target as you can make it.

### If the participant asks whether the practice counts

> No. The practice is only for familiarization. The main task comes after the practice.

## 19. Facilitator Quick Reference

- Study type: `within-subject`
- Condition start: condition title/explanation, `triple tap to continue`, phone QR scan, HoloLens says `Phone QR detected.`, match posture image, then `triple tap to start`
- After condition start: HoloLens says `Triple tap detected. Hold still.`, and participant should hold still briefly
- Task start: task title/explanation, `triple tap to continue`, optional practice, then `Main Placement/Rotation/Scaling Task Starts Now`
- Main submit action: `triple tap`
- Current workflow per condition: `Placement -> Rotation -> Scaling`, with `5 tools` per phase
- `Participant ID` should include a number if the Latin square sequence is linked to the participant ID
