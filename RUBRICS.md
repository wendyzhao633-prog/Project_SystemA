# RUBRICS.md

These rubrics are the source of truth for pairwise judging.

## Important scoring rule
- `D1`-`D5` are **quality** dimensions.
- `D6` is a **mechanism** dimension.
- `winner_overall` must be based on `D1`-`D5` only.
- `D6` must be reported separately and must not be merged into the overall decision.

For each dimension, the judge compares **Response A** and **Response B** and outputs one of:

- `A`
- `B`
- `tie`

---

## D1 Emotional Attunement & Calibration
**Definition:**
Whether the assistant correctly captures the user's emotional state **or the absence of strong emotion**, and matches the **intensity** appropriately without overreacting or underreacting.

### 1-5 Anchors
- **5**: Correctly captures the dominant emotion (or recognizes that the query is mainly task-focused / low-emotion); tone and intensity are well calibrated.
- **4**: Generally aligned, but slightly generic or a little over/understated.
- **3**: Polite and acceptable, but emotionally generic; could fit many unrelated cases.
- **2**: Noticeable mismatch in emotion or intensity; overreads or ignores the user's state.
- **1**: Dismissive, invalidating, or escalates the emotion.

---

## D2 Emotion-to-Response Strategy Fit
**Definition:**
Whether the assistant converts the emotional state into the **right response strategy**, not just the right tone.

### 1-5 Anchors
- **5**: Strategy clearly matches both emotion and task.
- **4**: Mostly appropriate, with a small mismatch in pacing, tone, or action style.
- **3**: Reasonable generic help, but little evidence of emotion-specific strategy.
- **2**: Strategy is poorly matched to the state.
- **1**: Strategy worsens the interaction or adds pressure.

Examples:
- anxious / fearful -> steadier wording, clearer priorities, reassurance
- angry -> validate frustration, avoid escalation, channel toward constructive action
- sad / low-energy -> gentler activation, smaller next step
- happy / surprise -> warm and energized, but still task-useful
- calm / neutral -> efficient and not over-emotional

---

## D3 Task Usefulness & Correctness
**Definition:**
Whether the response genuinely advances the user's actual goal, with content that is useful, actionable, and sufficiently correct for the scope of the request.

### 1-5 Anchors
- **5**: Directly moves the task forward with concrete and usable output.
- **4**: Helpful and relevant, but with minor gaps in detail, precision, or actionability.
- **3**: Some useful content, but generic or only partially advances the task.
- **2**: Weak progress; suggestions are vague, hard to execute, or somewhat mismatched.
- **1**: Off-topic, misleading, or practically unusable.

---

## D4 Clarity & Cognitive Load Fit
**Definition:**
Whether the response is easy to follow, well structured, and sized appropriately for the user's likely cognitive bandwidth.

### 1-5 Anchors
- **5**: Well structured, clearly prioritized, easy to act on, with the right level of detail for this case.
- **4**: Clear overall, but slightly wordy or a bit under-structured.
- **3**: Understandable, but lacks prioritization or feels somewhat scattered.
- **2**: Dense, rambling, or hard to execute.
- **1**: Confusing or poorly organized.

---

## D5 Tone Safety, Respect & Risk Calibration
**Definition:**
Whether the response is respectful, non-judgmental, and appropriately cautious in higher-stakes contexts; it should not shame, pressure, or give reckless reassurance.

### 1-5 Anchors
- **5**: Supportive and respectful; no blame or pressure; shows appropriate caution when stakes are higher.
- **4**: Generally safe, with only slight awkwardness or mild bluntness.
- **3**: Neutral and harmless.
- **2**: Some blame, pressure, dismissiveness, or overconfidence.
- **1**: Shaming, intimidating, or clearly unsafe / harmful.

---

## D6 Emotion Signal Utilization
**Definition:**
Whether the assistant shows **behavioral evidence** that it actually used the injected emotion signal (`valence/arousal` or `emotion_label`) to adjust acknowledgement, pacing, wording, and priorities, instead of giving a generic answer.

### 1-5 Anchors
- **5**: Clear evidence of signal use; response behavior is meaningfully adapted to the injected signal.
- **4**: Uses the signal, but somewhat shallowly or inconsistently.
- **3**: Slight personalization, but the response could probably have been produced without the signal.
- **2**: Mostly generic; little visible evidence of signal use.
- **1**: Contradicts, fabricates, or misuses the signal.

Examples:
- high arousal -> shorter, steadier, more prioritized
- low-arousal sadness -> gentler activation, smaller next action
- positive high arousal -> more energetic, celebratory but still focused
- calm / neutral -> less over-emotional, more direct

---

## Judge-side hard rules
The judge prompt must enforce these rules:

1. Do not reward longer responses just for being longer.
2. Do not reward explicit emotion words by themselves.
3. Do not punish concise responses if they are better calibrated.
4. Use `D1`-`D5` for overall judgment only.
5. Use `D6` as a separate mechanism judgment.
6. If the difference is small, choose `tie`.
