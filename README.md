# Aigile  
### *The Intelligent Automation of the Scrum Workflow*
<p align="center">
<!--   <img width="1500" height="1500" alt="Untitled-1 co2py" src="https://github.com/user-attachments/assets/60e7f2e5-070e-4aa3-a755-75e3ed38e705" /> -->

  <img src="https://github.com/user-attachments/assets/60e7f2e5-070e-4aa3-a755-75e3ed38e705" alt="Aigile Logo" width="200" height="3000" />
</p>

**Aigile** is an advanced, AI-powered platform designed to automate and enhance key activities within the Scrum framework.  
Developed as a graduation project at the Faculty of Engineering, Cairo University, this system integrates directly with **Jira** to streamline workflows, reduce manual effort, and introduce **data-driven intelligence** into agile project management.

## 📚 Table of Contents

- [🚀 About the Project](#-about-the-project)
- [✨ Key Features & Modules](#-key-features--modules)
  - [1. Backlog & Acceptance Criteria Generation](#1-backlog--acceptance-criteria-generation)
  - [2. Intelligent Story Point Estimation](#2-intelligent-story-point-estimation)
  - [3. Adaptive Task Assignment](#3-adaptive-task-assignment)
  - [4. Sprint Review Summarization](#4-sprint-review-summarization)
- [🏛️ System Architecture](#️-system-architecture)
- [🛠️ Technology Stack](#️-technology-stack)
- [🎬 Demo](#-demo)
- [📄 Research & Contribution](#-research--contribution)
- [👥 Team](#-team)
- [🙏 Acknowledgments](#-acknowledgments)

## 🚀 About the Project

The Agile methodology, while effective, relies on manual, time-consuming tasks like writing user stories, estimating effort, assigning tasks, and documenting meetings. These processes are often subjective and prone to human bias, leading to inefficiencies and inaccurate planning.

Aigile addresses these challenges by:

- **Automating Tedious Tasks**: Freeing up Product Owners and Scrum Masters to focus on strategy rather than administration.
- **Improving Accuracy**: Using machine learning to provide consistent, data-driven estimations and task assignments.
- **Enhancing Efficiency**: Accelerating core Scrum ceremonies like sprint planning and backlog refinement.
- **Providing Seamless Integration**: Embedding powerful AI tools directly into the Jira environment where teams already work.

The project is structured as a suite of five integrated Jira plugins and two standalone GUI applications, powered by a hybrid backend of a Flask server and serverless AWS infrastructure.

## ✨ Key Features & Modules

Aigile is composed of four intelligent modules, each targeting a critical part of the Scrum lifecycle.

### 1. Backlog & Acceptance Criteria Generation

This module transforms raw, high-level project requirements into a well-structured, prioritized product backlog.

**Workflow:**

- **Requirement Input**: A user inputs unstructured project requirements into the Aigile Jira plugin.
- **AI-Powered Generation**: The system uses the Llama-3.3-70B-Versatile LLM to generate a list of user stories, each conforming to the standard "As a..., I want to..., so that..." format.
- **Multi-Agent Prioritization**: To score each story, the system simulates a "100-dollar method" discussion between three AI agents representing a Product Owner, a Developer, and a QA Analyst, ensuring a balanced priority based on business value, technical feasibility, and quality assurance.
- **Acceptance Criteria**: For any generated user story, the system can then generate detailed acceptance criteria, ensuring each task is well-defined and testable before development begins.

### 2. Intelligent Story Point Estimation

This module provides accurate and adaptive effort estimation for user stories, tackling one of the most challenging aspects of agile planning. It was developed in three distinct phases to address real-world complexities.

**Workflow & Approaches:**

#### Phase 1: Cross-Project Estimation (Initial Exploration)

- **Goal**: To create a general model that works without any project-specific history.
- **Process**: We tested various models (SVM, SVR, LSTM) and text embeddings (BERT, SBERT). The key innovation was using an LLM to decompose complex user stories into smaller, more predictable subtasks. The final estimation is an average of the points predicted for each subtask.
- **Result**: This subtask decomposition approach significantly improved accuracy and provided better explainability.

#### Phase 2: Project-Specific Estimation (With Historical Context)

- **Goal**: To leverage a project's own historical data for highly accurate, context-aware predictions.
- **Process**: Our flagship model, the Enhanced MLP, finds the most similar user stories from the project's history. It then combines the new story's text embedding with a "context vector" from past stories and statistical features (mean, median, max story points) to make a highly informed prediction.
- **Result**: This method proved extremely effective for teams with existing data, with our FastText + SVM model achieving a low MAE of 1.245.

#### Phase 3: Incremental Learning (The Final, Hybrid Approach)

- **Goal**: To solve the "cold start" problem for new projects and adapt to changes in ongoing ones.
- **Process**: The system starts with a model trained on a general dataset of ~39 projects. As a team completes sprints, the model is incrementally updated with a batch of new data using the partial_fit method of the PassiveAgressive model. This "test-then-train" cycle keeps the model constantly adapting.
- **Result**: This approach provides the best of both worlds: it gives reliable estimates for new projects and becomes progressively more accurate as it learns a team's specific context. It achieves this with incredible efficiency, updating in ~0.02 seconds compared to hours for a full retrain.

### 3. Adaptive Task Assignment

This module recommends the most suitable developer for a given task, learning and adapting to team members' evolving skills and expertise over time.

**Workflow:**

- **Initial Training (Offline)**: The system is first trained on a project's historical issue data to learn which developers typically handle which types of tasks.
- **Online Learning (Real-Time)**: For each new task, the model recommends the top 3 developers. Once a project manager makes the final assignment, the model learns from this decision using the learn_one method from the river library.
- **Concept Drift & Recency**: The model uses an AdaBoost ensemble with ADWIN to detect "concept drift" (e.g., when a developer learns a new skill). It also uses a decay-based recency heuristic, giving more weight to recent assignments to stay current with the team's dynamic.
- **Result**: This online learning system achieved a peak accuracy of 74.7%, significantly outperforming static models.

### 4. Sprint Review Summarization

This module provides automated summaries of sprint review meeting transcripts, ensuring key decisions and action items are captured efficiently.

**Workflow:**

- **Transcript Input**: A user provides the text transcript of a meeting.
- **Dual Summarization Approach**:
  - **Extractive Summary**: Uses an enhanced TF-IDF and TextRank approach to pull the most critical sentences verbatim. The model gives extra weight to sentences containing agile keywords (e.g., "completed," "blocked") and those spoken by non-lead developers to capture the team's voice.
  - **Abstractive Summary**: Uses a fine-tuned BART model (trained on the SAMSum dialogue dataset) to generate a fluent, human-like narrative summary that synthesizes the key points of the conversation.

## 🏛️ System Architecture

Aigile is built on a distributed, microservices-inspired architecture designed for scalability and seamless integration.

- **Frontend**: A suite of five Jira plugins built with React and Atlassian Forge UI, providing a native experience. Two standalone GUI applications built with Python handle offline tasks.
- **Backend**: A hybrid system:
  - **Flask Server (on PythonAnywhere)**: Handles generative AI tasks (story/criteria generation) that require access to the powerful Llama-3.3 LLM via the Groq API.
  - **AWS Serverless Infrastructure**: A highly scalable and low-latency setup for predictive models.
    - API Gateway routes requests from the Jira plugins.
    - Lambda Functions host the story point and task assignment models.
    - S3 provides persistent, secure storage for the trained machine learning models.

## 🛠️ Technology Stack

- **AI & Machine Learning**: Python, PyTorch, TensorFlow, Scikit-learn, Hugging Face Transformers, River (Online Learning), NLTK, spaCy
- **Backend**: Flask, PythonAnywhere, AWS (Lambda, S3, API Gateway)
- **Frontend**: React, Atlassian Forge UI, JavaScript
- **APIs & Data**: Jira REST API, Groq API (for Llama-3.3)
- **Databases & Datasets**: TAWOS Dataset, Apache Jira Issue Tracking Dataset

## 🎬 Demo

[This is where you can embed a link to your project demo video. You can upload it to YouTube or another video hosting service.]

**[Link to Demo Video Here]**

## 📄 Research & Contribution

A key innovation of this project is the development of a novel incremental learning approach for story point estimation. This methodology addresses the critical "cold start" and concept drift problems in agile environments.

A research paper based on this work has been submitted for publication:

**"An Incremental Learning Approach for Realistic Story Point Prediction in Agile Frameworks"**

This paper details the model's architecture, experimental setup, and results, demonstrating its superior performance and efficiency compared to traditional static models.

## 👥 Team

<table>
  <tr>
    <td align="center">
    <a href="https://github.com/Menna-Ahmed7" target="_black">
    <img src="https://avatars.githubusercontent.com/u/110634473?v=4" width="150px;" alt="https://github.com/Menna-Ahmed7"/>
    <br />
    <sub><b>Mennatallah Ahmed</b></sub></a>
    </td>
    <td align="center">
    <a href="https://github.com/MostafaBinHani" target="_black">
    <img src="https://avatars.githubusercontent.com/u/119853216?v=4" width="150px;" alt="https://github.com/MostafaBinHani"/>
    <br />
    <sub><b>Mostafa Hani</b></sub></a>
    </td>
    <td align="center">
    <a href="https://github.com/MohammadAlomar8" target="_black">
    <img src="https://avatars.githubusercontent.com/u/119791309?v=4" width="150px;" alt="https://github.com/MohammadAlomar8"/>
    <br />
    <sub><b>Mohammed Alomar</b></sub></a>
    </td>
    <td align="center">
    <a href="https://github.com/mou-code" target="_black">
    <img src="https://avatars.githubusercontent.com/u/123744354?v=4" width="150px;" alt="https://github.com/mou-code"/>
    <br />
    <sub><b>Moustafa Mohammed</b></sub></a>
    </td>
  </tr>
 </table>


**Supervised by:** Dr. Ahmed Darwish

## 🙏 Acknowledgments

We extend our deepest gratitude to our supervisor, Dr. Ahmed Darwish, for his invaluable guidance and mentorship. We also thank the 30 anonymous Agile practitioners who participated in our surveys, providing crucial feedback for validating our work.
