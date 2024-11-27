# RAG-Based Talent Search and Recommendation System

<img src="[https://github.com/user-attachments/assets/69fef935-2676-48cf-9ef7-66cd85cad565](https://github.com/user-attachments/assets/6b5905f1-5e4c-49b6-a413-27a53fb6cf8f)" alt="image" style="width:800px;">

## Project Overview
This project aims to develop a "RAG-Based Talent Search and Recommendation System" to streamline corporate recruitment processes and quickly identify the most suitable candidates. The project utilizes Retrieval-Augmented Generation (RAG) technology to efficiently search and summarize a vast number of resumes and cover letters, providing companies with the best-fit candidates.

Currently, the recruitment environment is highly inefficient, requiring manual review of a massive volume of documents. This project aims to solve these issues by using state-of-the-art NLP technology, specifically RAG, to automate the document search and summary process, thereby maximizing the efficiency of recruitment processes.

## Project Background and Need
Modern recruitment is highly competitive, and the speed and accuracy of talent acquisition determine corporate competitiveness. Reviewing numerous resumes and cover letters manually takes significant time and lacks an objective and systematic evaluation system. To address these issues, the RAG-based Talent Search and Recommendation System has been introduced. RAG technology searches external knowledge bases and summarizes recent information, offering greater accuracy through contextual analysis rather than traditional keyword-based searches.

## Key Features
### 1. **Introduction of RAG Technology**
- Supports searching and summarizing the latest information using external knowledge bases. This ensures greater accuracy and utility compared to existing NLP technologies.

### 2. **Automated Scoring and Ranking**
- Automatically evaluates candidate suitability based on Job Description (JD), experience, qualifications, and skills. This provides an objective basis for evaluating and ranking candidates.

### 3. **Conversational Chatbot Interface**
- Provides a user-friendly chatbot interface that allows users to input their requirements easily and receive immediate results. This feature enhances user experience and accessibility.

## Differentiation Points and Visuals
1. **Contextual Analysis-Based Search**
   - Unlike existing similar services that rely solely on keyword matching, this system provides more accurate recommendations by analyzing context. The image below highlights this differentiation.


2. **User-Customized Recommendation System**
   - Customizes job postings or career maps based on the applicant's resume information. This helps save time and enhances efficiency for both companies and applicants.


3. **Urgent Talent Search Support**
   - Supports companies in urgently finding candidates for specific roles, providing summaries and visual score distributions for each recommended candidate.


## Technology Stack and Development Approach
This project is built using FastAPI for the backend and HTML, CSS, JavaScript for the frontend. Chroma DB and Pydantic Schema were utilized for database management, responsible for data validation and metadata processing. The Agile development methodology was adopted to iteratively improve functionality based on customer feedback.

### Technologies Used
- **RAG (Retrieval-Augmented Generation)**: A technology that generates responses using LLM by referring to external documents, improving reliability for up-to-date information.
- **Prompt Engineering**: Used Langchain's PromptTemplate and SystemMessage to ensure the system returns optimal values.
- **Embedding Model**: Used the KR-SBERT model for efficient vectorization of sentence meanings, enabling effective searching of candidate resumes.

## Project Development Timeline
- **November 7**: Started initial model development
- **November 14**: Improved evaluation method to reduce dependence on retrievers
- **November 18**: Introduced multi-level evaluation method to enhance evaluation precision
- **November 25**: Finalized the model by introducing RP (Retrieval Point) and FP (Fundamental Point) to improve score reliability and reduce computation.

## System Workflow
### 1. **Upload Workflow**
- PDF Upload -> Conversion to Document -> Metadata Extraction and Update -> Storage in Vector DB (ChromaDB)

   ![Upload Workflow](./static/upload_workflow.png)

### 2. **Search Workflow**
- JD Input -> Prompt Data Processing -> Information Retrieval -> Score Calculation (TP = RP + FP) -> Provide Recommended Candidate Rankings

   ![Search Workflow](./static/search_workflow.png)

## Project Outcome and Future Development
This project has significantly improved the inefficiencies of traditional recruitment processes, successfully enhancing recommendation accuracy and evaluation reliability. Future directions include multi-language support, adding detailed evaluation functions for more complex roles, and implementing more sophisticated user-customized recommendations.

Despite the team being composed of non-majors, the project was successfully carried out as a full-stack development, with a focus on applying the latest technology trends to improve project completeness. As a result, we achieved both enhanced technical skills and strengthened problem-solving capabilities.



# Citation

"""
This project uses KR-SBERT: A Pre-trained Korean-specific Sentence-BERT model.
For more information, visit: https://github.com/snunlp/KR-SBERT

If you use this model in your Project, please cite:
@misc{kr-sbert,
  author = {Park, Suzi and Hyopil Shin},
  title = {KR-SBERT: A Pre-trained Korean-specific Sentence-BERT model},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/snunlp/KR-SBERT}}
}
"""
