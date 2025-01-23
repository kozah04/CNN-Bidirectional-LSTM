# AI Models for Institutions: Datasets Overview

This document outlines the datasets compiled for three AI applications designed for institutional use:

1. **Exam Proctoring**: Using computer vision to monitor online exams and detect cheating behaviors.
2. **Automated Grading**: Leveraging NLP and machine learning for efficient and consistent grading of subjective and objective answers.
3. **Plagiarism Detection**: Analyzing exam answers or assignments for originality and flagging potential plagiarism.

Below are the datasets, their roles in achieving project goals, and their limitations for each application.

---

## **1. Exam Proctoring**
**Goal**: Use computer vision to monitor online exams and detect behaviors such as:
- **Looking away** (gaze tracking).
- **Using unauthorized devices** (object detection).
- **Consulting external materials** (action recognition).

### **Datasets**:
1. **[GazeCapture](https://gazecapture.csail.mit.edu/)**
   - **Description**: A large-scale dataset for gaze tracking, collected from real-world environments. Registration is needed to access this and doesn't accept freemail accounts.
   - **How it Helps**: Enables training of models to detect gaze direction and identify if students are looking away from the screen.
   - **Limitations**: May require additional preprocessing and fine-tuning for exam-specific scenarios.

2. **[Columbia Gaze Dataset](https://ceal.cs.columbia.edu/columbiagaze/#project-dataset)**
   - **Description**: A smaller, controlled dataset with annotations for gaze direction under various head poses and lighting conditions.
   - **How it Helps**: Complements GazeCapture for controlled gaze tracking experiments.
   - **Limitations**: Limited diversity and scale compared to GazeCapture.

3. **[WIDER FACE](http://shuoyang1213.me/WIDERFACE/WiderFace_Results.html)**
   - **Description**: A dataset for face detection in diverse conditions (e.g., occlusions, poses, lighting).
   - **How it Helps**: Trains face detection models for robust tracking in varied exam environments.
   - **Limitations**: Does not include behavior-specific annotations like gaze or actions.

4. **[AVA Actions (v2.2)](https://research.google.com/ava/download.html)**
   - **Description**: A video dataset with spatiotemporal annotations for actions like "reading," "writing," or "talking on the phone."
   - **How it Helps**: Detects cheating behaviors such as consulting external materials or interacting with unauthorized devices.
   - **Limitations**: Not specifically tailored for exam scenarios, requiring fine-tuning.

5. **[DAISEE](https://datasets.activeloop.ai/docs/ml/datasets/daisee-dataset/)**
   - **Description**: Annotated videos for engagement detection, including gaze shifts and attention levels.
   - **How it Helps**: Identifies disengagement or suspicious activities during proctoring.
   - **Limitations**: Focused on engagement detection rather than explicit cheating behaviors.

---

## **2. Automated Grading**
**Goal**: Grade subjective (e.g., essays) and objective (e.g., multiple-choice) answers efficiently and consistently using NLP.

### **Datasets**:
1. **[RACE Dataset](https://huggingface.co/datasets/ehovy/race)**
   - **Description**: A reading comprehension dataset with both multiple-choice and free-text answers.
   - **How it Helps**: Provides training data for models to grade both objective and subjective responses.
   - **Limitations**: Focused on English comprehension, requiring domain-specific data for other subjects.

2. **[The Hewlett Foundation: Automated Essay Scoring](https://www.kaggle.com/competitions/asap-aes/data)**
   - **Description**: A dataset of student essays graded by human raters based on predefined rubrics.
   - **How it Helps**: Enables training of essay-grading models for subjective assessments.
   - **Limitations**: Restricted to essay-type responses and predefined scoring rubrics.

3. **[ASAP 2.0](https://www.kaggle.com/datasets/lburleigh/asap-2-0)**
   - **Description**: An extension of the ASAP dataset with additional essay samples and annotations.
   - **How it Helps**: Enhances robustness of essay-grading models.
   - **Limitations**: Similar limitations as the Hewlett dataset.

4. **[SQuAD (Stanford Question Answering Dataset)](https://rajpurkar.github.io/SQuAD-explorer/)**
   - **Description**: A dataset for machine reading comprehension with context-question-answer triples.
   - **How it Helps**: Trains models to evaluate free-text answers by matching them to reference answers.
   - **Limitations**: Focused on fact-based Q&A, less suited for subjective analysis.

5. **[MCTest Dataset](https://huggingface.co/datasets/sagnikrayc/mctest)**
   - **Description**: A multiple-choice reading comprehension dataset.
   - **How it Helps**: Trains models for objective grading tasks.
   - **Limitations**: Limited to English reading comprehension.

6. **[The ARC Dataset](https://leaderboard.allenai.org/arc/submissions/get-started)**
   - **Description**: A challenging science question-answering dataset with both multiple-choice and short-answer formats.
   - **How it Helps**: Focused on objective grading in STEM subjects.
   - **Limitations**: Narrowly focused on science questions.

---

## **3. Plagiarism Detection**
**Goal**: Analyze exam answers or uploaded assignments for originality and flag potential plagiarism.

### **Datasets**:
1. **[PAN Plagiarism Corpus](https://zenodo.org/records/3250095)**
   - **Description**: A comprehensive dataset with labeled plagiarism cases, including exact matches, paraphrased content, and obfuscated plagiarism.
   - **How it Helps**: Trains models to identify different levels of text similarity and detect plagiarized content.
   - **Limitations**: May not cover domain-specific plagiarism scenarios (e.g., technical subjects).

2. **[Microsoft Research Paraphrase Corpus (MSRP)](https://www.microsoft.com/en-us/download/details.aspx?id=52398)**
   - **Description**: A dataset of sentence pairs labeled as semantically equivalent or not.
   - **How it Helps**: Detects paraphrased plagiarism and semantically similar text.
   - **Limitations**: Focused on short text pairs, less effective for long documents.

3. **[Clough & Stevenson Corpus](https://ir.shef.ac.uk/cloughie/resources/plagiarism_corpus.html)**
   - **Description**: A dataset of student assignments with manually annotated plagiarism cases.
   - **How it Helps**: Focused on plagiarism detection in academic writing.
   - **Limitations**: Limited scale and diversity.

---

## **Note**
The datasets listed above can be accessed using the provided links. Please ensure compliance with individual dataset licenses when using them.

