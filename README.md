# RAG-Based Talent Search and Recommendation System

![image](https://github.com/user-attachments/assets/c0081eb6-1b22-4a82-80d6-ff55cff0b1f3)

## Project Overview
This project aims to develop a "RAG-Based Talent Search and Recommendation System" to streamline corporate recruitment processes and identify suitable candidates efficiently. The project uses Retrieval-Augmented Generation (RAG) technology to search and summarize numerous resumes and cover letters, helping companies find the best-fit candidates.

The system was designed with insights from an HR specialist to ensure it meets the needs of real-world recruitment. Their expertise helped create a practical solution aligned with the complexities of modern hiring.

Currently, the recruitment process is often inefficient, requiring manual review of a large volume of documents. This project aims to address these inefficiencies by automating document search and summary using state-of-the-art NLP technology, specifically RAG.

## Project Background and Need
Recruitment today is highly competitive, where the speed and accuracy of talent acquisition play a significant role. Reviewing resumes manually is time-consuming and lacks systematic evaluation. To overcome these challenges, the RAG-based Talent Search and Recommendation System was introduced. RAG technology allows for more precise contextual analysis compared to traditional keyword-based searches.

## System Workflow

![image](https://github.com/user-attachments/assets/dbca8908-6535-4443-baac-78d6363a90cf)

## Key Features
### 1. **RAG Technology**
- Searches and summarizes the latest information using external knowledge bases, offering greater accuracy compared to existing NLP technologies.

### 2. **Automated Scoring and Ranking**
- Evaluates candidate suitability based on Job Description (JD), experience, qualifications, and skills, providing an objective ranking.

   ![image](https://github.com/user-attachments/assets/d7d91f48-e240-44eb-8c2e-e348408be2bb)

### 3. **Conversational Chatbot Interface**
- Offers a chatbot interface that allows users to enter requirements and receive immediate results, improving accessibility and user experience.

## Differentiation Points and Visuals
1. **Contextual Analysis-Based Search**
   - Unlike keyword-only services, this system analyzes context for more accurate recommendations.

2. **Urgent Talent Search Support**
   - Quickly finds candidates for urgent roles, providing summaries and score distributions for recommended candidates.

## Technology Stack and Development Approach
The project uses FastAPI for the backend and HTML, CSS, JavaScript for the frontend. The following technologies were employed as part of the project's solution:

- **RAG (Retrieval-Augmented Generation)**: Combines document retrieval with LLM-based generation to answer queries based on external documents.
- **Pydantic Schema**: Used in combination with FastAPI to validate and manage the structure of resume data.
- **Prompt Engineering**: Utilized Langchain's PromptTemplate and SystemMessage to optimize LLM responses for extracting key details from prompts.
- **Embedding Model (KR-SBERT)**: Used for efficient vectorization of sentence meanings, enabling precise resume search and matching.
- **LLM Model (Llama-3-Korean-Bllossom-8B)**: An LLM specifically optimized for the Korean language, enhancing the AI capabilities of the system.

Chroma DB was used for database management, and an agile methodology was adopted to iteratively enhance functionality based on feedback.

### Technologies Used

- **RAG (Retrieval-Augmented Generation)**: Generates responses using LLM by referring to external documents, ensuring reliable information.
- **Prompt Engineering**: Langchain's PromptTemplate and SystemMessage were used for optimal system output.
- **Embedding Model**: KR-SBERT model for efficient vectorization of sentence meanings, aiding in effective resume search.

## Project Development Timeline
- **November 7**: Initial model development started
- **November 14**: Improved evaluation method to reduce dependency on retrievers
- **November 18**: Introduced multi-level evaluation for better precision
- **November 25**: Finalized the model with RP (Retrieval Point) and FP (Fundamental Point) to enhance reliability and reduce computation

## User Guide

### 1. **How to Upload a Document**

- Click the 'Select File' button and choose the file to upload from your device -> Once you see the uploaded file name in the UI, click the 'Upload PDF' button and wait until the popup window appears

   ![image](https://github.com/user-attachments/assets/a738a4f6-da20-4bd5-8884-8b0f2b6dd4e4)

### 2. **How to Search for Candidates**

- Enter the Job Description (JD) or simply type in your desired conditions as a prompt -> Wait for the output, the chatbot will provide a response along with detailed explanations on the side

   ![image](https://github.com/user-attachments/assets/d0a77911-f574-48aa-baac-9570665c46f1)


## Project Outcome and Future Development
The project improved inefficiencies in traditional recruitment, enhancing recommendation accuracy and reliability. Input from an HR specialist ensured the solution meets real-world needs.

Future improvements include multi-language support, adding evaluation features for complex roles, and implementing advanced user customization.

Despite being developed by a team of non-specialists, the project was completed successfully as a full-stack development, applying modern technology trends to improve the solution. The experience resulted in strengthened technical and problem-solving skills.

# Citation

```
This project uses KR-SBERT: A Pre-trained Korean-specific Sentence-BERT model.
For more information, visit: https://github.com/snunlp/KR-SBERT

If you use this model in your project, please cite:
@misc{kr-sbert,
  author = {Park, Suzi and Hyopil Shin},
  title = {KR-SBERT: A Pre-trained Korean-specific Sentence-BERT model},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/snunlp/KR-SBERT}}
}
```

