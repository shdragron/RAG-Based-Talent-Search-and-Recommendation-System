# utils.py
import os
from glob import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
# 변경된 Chroma import
from pydantic import ValidationError
from typing import List
import re
import logging
import math
from models import Person, JDMetadata
import shutil
import json
from datetime import datetime



# PDF 파일 저장
def save_uploaded_file(file, directory):
    file_location = os.path.join(directory, file.filename)
    with open(file_location, "wb") as f:
        f.write(file.file.read())
    return file_location

# 여러 개의 PDF 파일을 로드
def load_multiple_pdfs(directory):
    pdf_files = glob(os.path.join(directory, '**.pdf'))
    all_data = []
    all_text = ''
    
    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file)
        data = loader.load()
        all_data.extend(data)
     
    for i in range(0, len(all_data)):
        all_text += all_data[i].page_content
    all_text = ''.join(all_text.split('\n'))
    logging.info(f"Total number of documents loaded: {len(all_data)/3}")
    return all_data, all_text

# 메타데이터 추출
def extract_metadata_llm(all_text):
    resumes = re.split(r"이\s*력\s*서", all_text)[1:] # 첫 번째 항목은 빈 값이므로 제외

    prompt = PromptTemplate.from_template(
        """ Extract relevant information from the following text:
        
        TEXT: {text} \n
        AI: """
    )
    
    llm = OllamaFunctions(model="hf.co/MLP-KTLim/llama-3-Korean-Bllossom-8B-gguf-Q4_K_M", format="json", temperature=0.3)
    responses = []

    for i, resume_text in enumerate(resumes, start=1):
        runnable = prompt | llm.with_structured_output(schema=Person)
        try:
            response = runnable.invoke({"text": resume_text})
            responses.append(response)
            logging.info(f"Processed resume {i} successfully.")
        except ValidationError as e:
            logging.error(f"Validation failed for resume {i}: {e.json()}")
        except Exception as e:
            logging.error(f"An unexpected error occurred while processing resume {i}: {e}")

    return responses, resumes

# 메타데이터를 문서에 추가
def input_metadata(responses, resumes, all_data):
    pdf_docs = []
    split_pattern_1 = r'이.*력.*서'

    for i, doc in enumerate(all_data):
        split_text = re.search(split_pattern_1, doc.page_content)

        if split_text:
            person_metadata = {}  # 각 페이지별 메타데이터 저장
            try:
                response = responses[math.floor(i / 3)]
                for k, v in dict(response).items():
                    # None 값을 빈 문자열로 대체하거나 기본 값 할당
                    if v is None:
                        person_metadata[k] = ""
                    else:
                        person_metadata[k] = v
                logging.info(f"Print Meta_data {i}: {str(person_metadata)}")

            except Exception as e:
                logging.error(f"Error in metadata extraction for page {i}: {str(e)}")

            doc.metadata.update(person_metadata)
            logging.info(f"Print Meta_data_2 {i}: {str(doc.metadata)}")
        
        else:
            doc.metadata.update(person_metadata)

        pdf_docs.append(doc)

    return pdf_docs

def split_and_chunk_documents(all_data):
    tokenizer = AutoTokenizer.from_pretrained('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size=200,
        chunk_overlap=10,
    )

    # 분할된 문서 저장할 리스트 초기화
    split_docs = []

    # 각 문서에 대해 청크화 진행
    for original_doc in all_data:
        # original_doc가 Document 객체인지 확인
        if not hasattr(original_doc, 'page_content'):
            logging.error("Expected a Document object but got a different type.")
            continue

        # 각 청크의 메타데이터를 유지하면서 청크화
        chunks = text_splitter.split_documents([original_doc])
        for chunk in chunks:
            chunk.metadata = original_doc.metadata  # 메타데이터 상속
            logging.info(f"!! Chunked metadata: {chunk.metadata}...")
            split_docs.append(chunk)

    logging.info(f"Number of split documents: {len(split_docs)}")
    return split_docs




# 벡터스토어에 임베딩 저장
def Embedding(split_docs, vectorstore):
    final_docs = []
    for doc in split_docs:
        # 각 청크의 metadata에서 person_korean_name 키를 가져옴
        person_name = doc.metadata['person_korean_name']
        doc.page_content = f"###'{person_name}' 지원자를 추천합니다.\n\n" + re.sub(r'(?<!\.)\n', ' ', doc.page_content)

        # 메타데이터 간단하게 변환
        simplified_metadata = {}
        for key, value in doc.metadata.items():
            if isinstance(value, (str, int, float, bool)):
                simplified_metadata[key] = value
            elif isinstance(value, dict):
                # dict는 문자열로 변환하여 저장
                simplified_metadata[key] = json.dumps(value, ensure_ascii=False)
            else:
                # 다른 자료형은 문자열로 변환
                simplified_metadata[key] = str(value)

        # 수정된 메타데이터 적용
        doc.metadata = simplified_metadata

        final_docs.append(doc)
    
    vectorstore.add_documents(final_docs)
    return vectorstore



# PDF 파일을 백업 폴더로 이동
def backup_processed_pdfs(directory, backup_directory="pdf_data_backup"):
    # 백업 디렉토리가 존재하지 않으면 생성합니다.
    if not os.path.exists(backup_directory):
        os.makedirs(backup_directory)

    # 'directory'에서 모든 PDF 파일을 backup_directory로 이동합니다.
    pdf_files = glob(os.path.join(directory, '*.pdf'))
    for pdf_file in pdf_files:
        try:
            # 파일 이동
            shutil.move(pdf_file, os.path.join(backup_directory, os.path.basename(pdf_file)))
            logging.info(f"Moved file to backup: {os.path.basename(pdf_file)}")
        except Exception as e:
            logging.error(f"Error moving file {os.path.basename(pdf_file)}: {str(e)}")


from langchain.schema import SystemMessage, HumanMessage, AIMessage


def generate_answer_with_llm(top_docs, prompt):
    from langchain.chat_models import ChatOllama
    from langchain.schema import SystemMessage, HumanMessage, AIMessage

    # 문서에서 필요한 메타데이터를 추출하여 간단한 데이터 형태로 준비하는 함수
    def prepare_documents_for_llm(docs):
        processed_docs = []
        processed_candidates = set()  # 중복 후보자 제거를 위한 집합
        for idx, doc in enumerate(docs):
            # 메타데이터에서 필요한 값 추출
            metadata = doc.metadata
            person_name = metadata.get('person_korean_name', '이름 없음')
            if person_name in processed_candidates:
                continue  # 이미 처리된 후보자이면 건너뜀
            processed_candidates.add(person_name)
            university = metadata.get('person_university', '대학교 없음')
            major = metadata.get('person_major', '전공 없음')
            experience = metadata.get('person_experience', '경험 없음')
            TP_score = metadata.get('TP_score', '점수 없음')
            RP_score = metadata.get('RP_score', '점수 없음')
            FP_score = metadata.get('FP_score', '점수 없음')
            age_score = metadata.get('Age_score', 0)
            major_score = metadata.get('Major_score', 0)
            skill_score = metadata.get('Skill_score', 0)
            experience_score = metadata.get('Experience_score', 0)

            # 경력 정보를 JSON으로 파싱
            try:
                experience = json.loads(experience)
            except:
                pass  # 변환 실패 시 그대로 사용

            # 요약된 데이터를 리스트에 추가
            processed_doc = {
                "종합 점수": TP_score,
                "이름": person_name,
                "대학교": university,
                "전공": major,
                "경력": experience,
                "리트리버 점수": RP_score,
                "인재 점수": FP_score,
                "Age_score": age_score,
                "Major_score": major_score,
                "Skill_score": skill_score,
                "Experience_score": experience_score
            }
            logging.info(f"Prepared document {idx}: {processed_doc}")
            processed_docs.append(processed_doc)
        return processed_docs

    # 문서 데이터를 간단한 형태로 준비
    processed_docs = prepare_documents_for_llm(top_docs)
    formatted_docs = json.dumps(processed_docs, indent=2, ensure_ascii=False)
    logging.info(f"Formatted documents content: {formatted_docs[:100]}...")

    # LLM 모델 인스턴스화 (ChatOllama 사용)
    llm = ChatOllama(model="hf.co/MLP-KTLim/llama-3-Korean-Bllossom-8B-gguf-Q4_K_M", temperature=0.2)
    logging.info(f"LLM instance created. Model: {llm.model}, Temperature: {llm.temperature}")

    # 입력 데이터를 하나의 문자열로 생성
    instruction = (
        '주어진 이력서 데이터를 기반으로 후보자 목록을 생성하세요. '
        '각 후보자의 정보는 다음과 같은 형식으로 출력되어야 합니다:\n\n'
        '```json\n'
        '[\n'
        '  {\n'
        '    "종합 점수": 정수 또는 실수,\n'
        '    "리트리버 점수": 정수 또는 실수,\n'
        '    "인재 점수": 정수 또는 실수,\n'
        '    "이름": 문자열,\n'
        '    "대학교": 문자열,\n'
        '    "전공": 문자열,\n'
        '    "경력": 문자열 또는 객체 또는 배열,\n'
        '    "Age_score": 정수 또는 실수,\n'
        '    "Major_score": 정수 또는 실수,\n'
        '    "Skill_score": 정수 또는 실수,\n'
        '    "Experience_score": 정수 또는 실수\n'
        '  },\n'
        '  ...\n'
        ']\n'
        '```\n'
        '**주의: 오직 JSON 형식으로만 응답하고, 그 외의 텍스트는 포함하지 마세요.**'
    )

    input_data = f"이력서 데이터:\n{formatted_docs}"
    logging.info(f"Input data type: {type(input_data)}")
    logging.info(f"Input data content: {input_data[:100]}...")

    # 메시지 설정
    system_message = SystemMessage(content=instruction)
    user_message = HumanMessage(content=input_data)
    logging.info(f"System message type: {type(system_message)}")
    logging.info(f"User message type: {type(user_message)}")

    # LLM 호출 및 응답 처리
    try:
        logging.info("Starting LLM answer generation with strict JSON-only instruction...")
        # 메시지 리스트 전달
        response = llm.invoke([system_message, user_message])
        logging.info(f"Response type: {type(response)}")

        # 응답이 AIMessage 객체일 경우 content를 추출
        if isinstance(response, AIMessage):
            combined_output = response.content
        elif isinstance(response, str):
            combined_output = response
        else:
            raise TypeError(f"Unexpected response type: {type(response)}")

        logging.info(f"Combined output type: {type(combined_output)}")
        combined_output = combined_output.strip()
        logging.info(f"Combined output content: {combined_output[:100]}...")

        # LLM 출력이 코드 블록에 포함되어 있을 수 있으므로, 이를 제거합니다.
        json_string = combined_output
        if json_string.startswith("```"):
            json_string = json_string.strip("```")
            json_string = json_string.strip("json")
            json_string = json_string.strip()

        # JSON 파싱
        try:
            json_output = json.loads(json_string)
            logging.info(f"Parsed JSON output type: {type(json_output)}")
            logging.info(f"Successfully parsed JSON output: {json_output}")

            # 경력 필드가 JSON 형식의 문자열일 경우 딕셔너리로 변환
            for idx, item in enumerate(json_output):
                if "경력" in item and isinstance(item["경력"], str):
                    try:
                        parsed_experience = json.loads(item["경력"])
                        if isinstance(parsed_experience, (dict, list)):
                            item["경력"] = parsed_experience
                            logging.info(f"Parsed experience for item {idx}: {item['경력']}")
                        else:
                            logging.info(f"Keeping original experience as string for item {idx}")
                    except json.JSONDecodeError:
                        logging.warning(f"Failed to parse '경력' field for item {idx}: {item}")

            # JSON 문자열로 변환하여 반환
            return json.dumps(json_output, ensure_ascii=False)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse output as JSON: {e}")
            return "LLM이 유효한 JSON을 생성하지 못했습니다. 다시 시도해주세요."

    except Exception as e:
        # LLM 호출 실패에 대한 로그 기록 및 에러 메시지 반환
        logging.error(f"Error in LLM answer generation: {str(e)}")
        return "LLM이 응답을 생성하지 못했습니다. 다시 시도해주세요."
    
    
# FP 점수를 계산하는 함수
def calculate_fp_scores(pdf_docs, jd_data):
    from collections import defaultdict

    updated_docs = []

    # Group documents by 'person_korean_name' metadata
    grouped_docs_dict = defaultdict(list)
    for doc in pdf_docs:
        person_name = doc.metadata.get('person_korean_name', 'Unknown')
        grouped_docs_dict[person_name].append(doc)

    # For each group of documents (per person)
    for person_name, docs in grouped_docs_dict.items():
        # Combine the page contents
        combined_text = " ".join([doc.page_content for doc in docs])
        # Calculate FP scores using the combined text
        logging.info(f"jd_data: {jd_data}")
        fp_score, age_score, major_score, skill_score, experience_score = calculate_fp(combined_text, jd_data)
        # Assign the scores to each document in the group
        for doc in docs:
            doc.metadata['FP_score'] = fp_score
            doc.metadata['Age_score'] = age_score
            doc.metadata['Major_score'] = major_score
            doc.metadata['Skill_score'] = skill_score
            doc.metadata['Experience_score'] = experience_score
            updated_docs.append(doc)
        logging.info(f"Calculated scores for {person_name}: FP_score={fp_score}, Age_score={age_score}, Major_score={major_score}, Skill_score={skill_score}, Experience_score={experience_score}")
    return updated_docs



from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import PromptTemplate

def extract_jd_metadata_llm(jd_text) -> JDMetadata:
    parser = JsonOutputParser(schema=JDMetadata)
    
    format_instructions = """
    다음과 같은 JSON 형식으로 정보를 반환하세요.

    {
    "JD_age": 나이 (정수형 배열, 없을 경우 [] 또는 None),
    "JD_major": 전공 (전공 목록 (문자열 배열, 없을 경우 빈 배열 []),),
    "JD_keywords": 직무 키워드 목록 (문자열 배열, 없을 경우 빈 배열 []),
    "JD_career": 경력 년수 (정수형 배열, 없을 경우 [] 또는 None)
    }

    반드시 위의 JSON 형식으로만 응답하고, 그 외의 텍스트는 포함하지 마세요.
    """

    prompt = PromptTemplate(
        template="""
    다음 채용 공고에서 관련 정보를 추출하세요.
    {format_instructions}

    예시:
    채용 공고:
    "백엔드 개발자를 모집합니다. 요구 기술은 Python, Django, REST API입니다."

    예상 출력:
    {{
        "JD_age": [0,0],
        "JD_major": [],
        "JD_skills": ["Back-end","백앤드","개발","Python", "Django", "REST API"],
        "JD_career": [0,0]
    }}

    채용 공고:
    {jd_text}
    """,
        input_variables=["jd_text"],
        partial_variables={"format_instructions": format_instructions},
    )
    
    llm = ChatOllama(model="hf.co/MLP-KTLim/llama-3-Korean-Bllossom-8B-gguf-Q4_K_M", temperature=0.5)
    chain = prompt | llm
    
    try:
        raw_output = chain.invoke({"jd_text": jd_text})
        logging.info(f"LLM raw output: {raw_output}")
        
        # raw_output이 AIMessage 객체인 경우 content를 추출
        if isinstance(raw_output, AIMessage):
            json_string = raw_output.content
        elif isinstance(raw_output, str):
            json_string = raw_output
        else:
            logging.error(f"Unexpected type for raw_output: {type(raw_output)}")
            return None
        
        # LLM 출력이 코드 블록에 포함되어 있을 수 있으므로, 이를 제거합니다.
        json_string = json_string.strip()
        if json_string.startswith("```"):
            json_string = json_string.strip("```")
            json_string = json_string.strip("json")
            json_string = json_string.strip()
        
        # JSON 파싱
        try:
            json_data = json.loads(json_string)
        except json.JSONDecodeError as e:
            logging.error(f"JSON 파싱 오류: {str(e)}")
            return None
        
        # Pydantic 모델로 변환 (키워드 인자 사용)
        response = JDMetadata(**json_data)
        return response
    except Exception as e:
        logging.error(f"예상치 못한 오류 발생: {str(e)}")
        return None



# FP 계산 로직
def calculate_fp(resume_text, jd_data):
    
    # JD의 주요 항목에 대한 적합성 평가 함수 정의
    # JD의 주요 항목 추출 (나이, 학과, 기술, 경력 등)
    def is_age_appropriate(jd_age, resume_text):
        
        if jd_age is None:
            return False
        # llama3에서 가져온 나이와 이력서에서 제공된 나이 비교
        else:
            person_year = int(re.findall(r'(\d{2})\d{4}.*\d{6}', resume_text)[0])
            logging.info(f"Found person_year: {person_year}")

            # 연나이 계산
            if person_year <= 24:
                person_age = 24 - person_year
            else:
                person_age = 124 - person_year
            logging.info(f"Calculated person_age: {person_age}")
            logging.info(f"JD age: {jd_age}")
            
            try:
                if jd_age[0] == 0:
                    if person_age <= jd_age[1]:
                        return True
                    else:
                        return False
                elif len(jd_age) == 1:
                    logging.info(f"JD age: {jd_age[0]}, {type(jd_age[0])}")
                    logging.info(f"Person age: {person_age}, {type(person_age)}") 
                    if person_age >= jd_age[0]:
                        return True
                    else:
                        return False
                elif jd_age[1] == 0:
                    if person_age >= jd_age[0]:
                        return True
                    else:
                        return False
                else:
                    if person_age >= jd_age[0] and person_age <= jd_age[1]:
                        return True
                    else:
                        return False
            except Exception as e:
                logging.error(f"Error in age comparison: {str(e)}")
                return False

    def is_major_appropriate(jd_major, resume_text):
        if jd_major is None:
            return False
        # JD의 전공 키워드가 이력서에 있는지 확인
        for major in jd_major:
            if major in resume_text:
                return True
        else:
            return False

    def is_experience_appropriate(jd_experience, resume_text):
        try:
            if jd_experience is None or []:
                return False
            # llama3에서 가져온 나이와 이력서에서 제공된 나이 비교
            else:
                # 기간 추출을 위한 정규 표현식
                period_pattern = r'경력사항'
                period = re.split(period_pattern, resume_text)[1]
                period_pattern_2 = r'직무경험'
                period = re.split(period_pattern_2, period)[0]
                period_pattern_1 = r'(\d{4}\.\d{2})\s*~\s*(\d{4}\.\d{2})'
                matches = re.findall(period_pattern_1, period)

                # 경력 기간 계산
                total_months = 0

                for start_str, end_str in matches:
                    start_date = datetime.strptime(start_str, "%Y.%m")
                    end_date = datetime.strptime(end_str, "%Y.%m")
                    
                    # 두 날짜 사이의 개월 수 계산
                    months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1
                    total_months += months
                
                if jd_experience[0] == 0:
                    if total_months <= jd_experience[1]*12:
                        return True
                    else:
                        return False
                elif jd_experience[1] == 0:
                    if total_months >= jd_experience[0]*12:
                        return True
                    else:
                        return False
                else:
                    if total_months >= jd_experience[0]*12 and total_months <= jd_experience[1]*12:
                        return True
                    else:
                        return False
                
        except Exception as e:
                logging.error(f"Error in experience comparison: {str(e)}")
                return False
    
    def extract_matching_skills(jd_skills, resume_text):
        logging.info(f"JD skills: {jd_skills}")
        if not jd_skills:
            return []
        matching_skills = []
    
        for skill in jd_skills:
            if skill == '업무' or skill == '기술':
                logging.info(f"remove skill: {skill}")
            else:
                seperated_skill = skill.split(',')
                logging.info(f"Checking for skill: {skill}")
                for skill in seperated_skill:
                    if skill in resume_text:
                        matching_skills.append(skill)
        return matching_skills

    # JD의 주요 항목 추출 (나이, 학과, 기술, 경력 등)
    fp_score = 0
    weight_age = 0.1
    weight_major = 0.15
    weight_skill = 0.5
    weight_experience = 0.3
    age = 0
    major = 0
    experience = 0
    skill = 0
    
    logging.info(f"Checking JD data: {jd_data.JD_age}, {jd_data.JD_major}, {jd_data.JD_keywords}, {jd_data.JD_career}")

    # 나이 비교
    if is_age_appropriate(jd_data.JD_age, resume_text):
        age += weight_age * 100

    # 전공 비교
    if jd_data.JD_major and is_major_appropriate(jd_data.JD_major, resume_text):
        major += weight_major * 100

    # 스킬 비교
    matching_skills = extract_matching_skills(jd_data.JD_keywords, resume_text)
    skill += weight_skill * len(matching_skills) * 40  # 매칭된 기술당 20점

    # 경력 비교
    if jd_data.JD_career and is_experience_appropriate(jd_data.JD_career, resume_text):
        experience += weight_experience * 100

    fp_score = age + major + skill + experience

    return fp_score, age, major, skill, experience
    
def calculate_tp_scores(docs):
    # 문서에서 중복 없는 고유한 사람만 선택
    unique_docs = []
    seen_person_ids = set()

    for doc in docs:
        person_id = doc.metadata.get("person_korean_name")
        if person_id and person_id not in seen_person_ids:
            logging.info(f"Person other5 scores: {doc.metadata.get('FP_score', 0)}")
            
            fp_score = doc.metadata.get("FP_score", 0)
            rp_score = doc.metadata.get("RP_score", 0)  # 기본값 0 설정
            
            tp_score = rp_score + fp_score
            logging.info(f"Person other6 scores: {rp_score}, {tp_score}")

            # 메타데이터 업데이트
            doc.metadata["TP_score"] = tp_score

            # 고유 문서로 추가
            unique_docs.append(doc)
            seen_person_ids.add(person_id)

    return unique_docs