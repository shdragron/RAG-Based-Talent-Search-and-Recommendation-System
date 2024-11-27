# 사건에 대한 메타데이터 추출하는 Pydantic 스키마 정의

from typing import Union, Optional
from pydantic import BaseModel, Field
from langchain_huggingface import HuggingFaceEmbeddings

# 임베딩 모델 초기화
embeddings_model = HuggingFaceEmbeddings(model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS")

class Person(BaseModel):
    person_korean_name: Optional[str] = Field(None, description="이름이 무엇인가?")
    person_english_name: Optional[str] = Field(None, description="영문 이름이 무엇인가?")
    person_phone_number: Optional[str] = Field(None, description="휴대폰 번호 (e.g. 010-1234-5678)")
    person_birth: Optional[str] = Field(None, description="생년월일이 언제인가? (e.g. 1990-01-01)")
    person_university: Optional[str] = Field(None, description="출신대학교 (e.g. --대학교)")
    person_major: Optional[str] = Field(None, description="전공 (e.g. ---과)")
    person_experience: Optional[Union[str, dict, list[dict]]] = Field(
        None, description="전체 경력 정보를 문자열, 딕셔너리, 또는 딕셔너리의 리스트 형태로 받습니다."
    )
    # person_language: Optional[Union[str, dict]] = Field(
    #     None, description="외국어 능력 (예: 언어: 등급 또는 점수)"
    # )

class JDMetadata(BaseModel):
    JD_age:Optional[list[int]] = Field(None, description="나이 요구사항 (e.g. 30세 이상 -> [30,0], 20세 이상 40세 이하 -> [20,40], 40세 이하 -> [0,40]) / 없으면 None")
    JD_major: Optional[Union[str, list[str]]] = Field(None, description="전공 요구사항 (e.g. 컴퓨터공학, 경영학, 디자인) 찾으면 이름과 성향이 비슷한 학과를 한 개당 2개 이상 찾아줘 예를들어  컴퓨터 공학이면 컴퓨터학과, 소프트웨어학과 등등 데이터 증가해줘/ 없으면 []")
    JD_keywords: Optional[list[str]] = Field(None,  description="채용 공고에서 요구하는 기술 키워드 목록을 추출하 관련 키워드 데이터 증강하세요ㄴㄴ. (예: Python, Java, Spring, React 등) 없으면 빈 배열 []을 반환하세요.")
    JD_career: Optional[list[int]] = Field(None, description="경력 요구사항 (e.g. 경력 3년 이상 7년 이하 -> [3,7] ,경력 5년 이상 -> [5,0]), 3년 이하 -> [0,3] / 없으면 None")