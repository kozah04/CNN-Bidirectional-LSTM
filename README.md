# AI Models for Institutions: Datasets Overview

This document outlines the datasets compiled for three AI applications designed for institutional use:

1. **Exam Proctoring**: Using computer vision to monitor online exams and detect cheating behaviors.
2. **Automated Grading**: Leveraging NLP and machine learning for efficient and consistent grading of subjective and objective answers.
3. **Plagiarism Detection**: Analyzing exam answers or assignments for originality and flagging potential plagiarism.

Below are the dataset descriptions and how they can be used in achieving the project goals with their specific usage during training.

---

## **1. Exam Proctoring**
**Goal**: Use computer vision to monitor online exams and detect behaviors such as:
- **Looking away** (gaze tracking).
- **Using unauthorized devices** (object detection).
- **Consulting external materials** (action recognition).

### **Datasets and Usage**:
1. **[GazeCapture](https://gazecapture.csail.mit.edu/)**
   The GazeCapture dataset is a large-scale dataset designed for gaze tracking and collected from real-world environments. It provides diverse gaze data useful for training models to detect if a student is looking away from the screen during an online exam. Preprocessing is required to normalize gaze directions and align annotations with facial bounding boxes. This limitation can be addressed by combining it with smaller, controlled datasets like Columbia Gaze.

2. **[Columbia Gaze Dataset](https://ceal.cs.columbia.edu/columbiagaze/#project-dataset)**
   The Columbia Gaze dataset is a smaller, controlled dataset that provides annotations for gaze direction under different head poses and lighting conditions. It is particularly useful for fine-tuning gaze detection models trained on larger datasets like GazeCapture, improving accuracy in controlled environments despite its limited diversity.

3. **[WIDER FACE](http://shuoyang1213.me/WIDERFACE/WiderFace_Results.html)**
   WIDER FACE is a face detection dataset that includes images with diverse conditions, such as occlusions, poses, and varied lighting. It can be used for training models to accurately detect and track faces during online exams, ensuring the reliability of subsequent gaze or action detection tasks. Although it does not include annotations for gaze or behavior detection, it can be used as a pretraining dataset alongside GazeCapture or AVA Actions.

4. **[AVA Actions (v2.2)](https://research.google.com/ava/download.html)**
   The AVA Actions dataset provides spatiotemporal annotations for various human actions, such as "reading," "writing," and "talking on the phone." It is highly useful for training models to detect cheating behaviors like consulting notes or interacting with unauthorized devices during exams. However, fine-tuning is needed to align with exam-specific cheating actions, which can be addressed by supplementing it with custom annotations.

5. **[DAISEE](https://datasets.activeloop.ai/docs/ml/datasets/daisee-dataset/)**
   DAISEE is a dataset focused on engagement detection, including gaze shifts and attention levels. It is useful for identifying disengagement or suspicious behaviors during exams when combined with gaze-tracking models. Since it focuses on engagement rather than explicit cheating behaviors, it works best when paired with datasets like GazeCapture and AVA Actions for a comprehensive solution.

---

## **2. Automated Grading**
**Goal**: Grade subjective (e.g., essays) and objective (e.g., multiple-choice) answers efficiently and consistently using NLP.

### **Datasets and Usage**:
1. **[RACE Dataset](https://huggingface.co/datasets/ehovy/race)**
   The RACE dataset contains reading comprehension questions with both multiple-choice and free-text answers. It is highly relevant for training models to grade both objective and subjective answers, with multiple-choice questions supporting answer key matching and free-text responses enabling semantic similarity scoring. To broaden its applicability to other subjects, it should be supplemented with domain-specific datasets.

2. **[The Hewlett Foundation: Automated Essay Scoring](https://www.kaggle.com/competitions/asap-aes/data)**
   This dataset includes essays graded by human raters based on predefined rubrics, making it ideal for training essay-grading models to predict rubric-based scores for subjective responses. Its focus on essays restricts its application to other response types, which can be mitigated by combining it with datasets like SQuAD for grading short free-text answers.

3. **[ASAP 2.0](https://www.kaggle.com/datasets/lburleigh/asap-2-0)**
   ASAP 2.0 expands on the original ASAP dataset with additional essay samples and annotations, improving model robustness across diverse writing styles and prompts. While it overlaps with the Hewlett dataset, using both datasets together addresses limitations in scale and diversity.

4. **[SQuAD (Stanford Question Answering Dataset)](https://rajpurkar.github.io/SQuAD-explorer/)**
   SQuAD is a machine reading comprehension dataset that includes context-question-answer triples. It is highly effective for evaluating free-text answers by matching them to reference answers. However, its focus on fact-based Q&A limits its suitability for creative or open-ended responses, which can be addressed by combining it with essay datasets for subjective tasks.

5. **[MCTest Dataset](https://huggingface.co/datasets/sagnikrayc/mctest)**
   MCTest is a multiple-choice reading comprehension dataset. It is useful for training models to grade objective answers by identifying correct choices from distractors. Its simplicity and focus on reading comprehension limit its applicability to complex or domain-specific questions, which can be mitigated by using it alongside datasets like ARC.

6. **[The ARC Dataset](https://leaderboard.allenai.org/arc/submissions/get-started)**
   The ARC dataset focuses on science-related questions and includes both multiple-choice and short-answer formats. It is highly relevant for STEM-related grading tasks, particularly for reasoning-based questions. Combining it with datasets like RACE can help address its narrow focus on science.

---

## **3. Plagiarism Detection**
**Goal**: Analyze exam answers or uploaded assignments for originality and flag potential plagiarism.

### **Datasets and Usage**:
1. **[PAN Plagiarism Corpus](https://zenodo.org/records/3250095)**
   The PAN Plagiarism Corpus includes labeled plagiarism cases, such as exact matches, paraphrased content, and obfuscated plagiarism. It is highly suitable for training models to detect various levels of text similarity. While the dataset may lack domain-specific examples, such as technical documents, this can be addressed by augmenting it with custom data from specific domains.

2. **[Microsoft Research Paraphrase Corpus (MSRP)](https://www.microsoft.com/en-us/download/details.aspx?id=52398)**
   MSRP contains sentence pairs labeled as semantically equivalent or not, making it ideal for detecting paraphrased plagiarism. Its focus on sentence-level paraphrases limits its applicability to long-form documents, but this can be mitigated by combining it with datasets like PAN for document-level detection.

---
