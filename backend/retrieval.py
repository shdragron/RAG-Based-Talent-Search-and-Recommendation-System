import logging
from models import Person

def retrieve_candidates(prompt: str, vectorstore, top_k: int = 10):
    try:
        # 벡터스토어에서 모든 문서를 가져옴
        docs = vectorstore.similarity_search(prompt, k=top_k)

        logging.info(f"Number of documents retrieved: {len(docs)}")
        
        # 중복 점수 계산용 데이터 초기화
        person_score_map = {}
        person_metadata_map = {}
        
        # Pydantic 필드 접근 방식 설정
        try:
            person_fields = Person.model_fields  # Pydantic v2
        except AttributeError:
            person_fields = Person.__fields__  # Pydantic v1

        # 각 문서의 메타데이터 처리
        for doc in docs:
            try:
                # doc.metadata를 Pydantic 모델로 변환
                person_metadata = {key: value for key, value in doc.metadata.items() if key in person_fields}
                logging.info(f"Person metadata: {person_metadata}")

                if not person_metadata:
                    logging.warning("Person metadata is empty. Skipping this document.")
                    continue

                # Person 모델 초기화
                person = Person(**person_metadata)
                logging.info(f"Created Person model: {person}")

                # 중복 점수 처리
                person_id = person.person_korean_name
                if not person_id:
                    logging.warning("Person ID (name) is missing. Skipping this document.")
                    continue

                if person_id not in person_score_map:
                    person_score_map[person_id] = {
                        "rp_score": 0,
                        "fp_score": doc.metadata.get("FP_score", 0)
                    }
                    person_metadata_map[person_id] = person_metadata
                
                # 중복 횟수에 따른 점수 추가
                person_score_map[person_id]["rp_score"] += 3  # 한 번 중복 시 3점 증가
                logging.info(f"Person scores: {person_score_map[person_id]}")

            except Exception as e:
                logging.error(f"Error creating Person model: {str(e)}")

        # 문서에서 중복 없는 고유한 사람만 선택
        unique_docs = []
        seen_person_ids = set()

        for doc in docs:
            person_id = doc.metadata.get("person_korean_name")
            if person_id and person_id not in seen_person_ids:
                if person_id in person_score_map:
                    logging.info(f"Person other1 scores: {person_score_map[person_id]['rp_score']}")
                    
                    # 'RP'와 'TP' 점수 계산
                    rp_score = person_score_map[person_id]["rp_score"]
                    logging.info(f"Person other3 scores: {rp_score}")

                    # 메타데이터 업데이트
                    doc.metadata["RP_score"] = rp_score

                    # 고유 문서로 추가
                    unique_docs.append(doc)
                    seen_person_ids.add(person_id)
                else:
                    logging.warning(f"Person ID {person_id} not found in person_score_map")

            # 최대 5개의 다른 사람의 문서만 선택
            if len(unique_docs) >= top_k:
                break

    except Exception as e:
        logging.error(f"Error in retrieving candidates: {str(e)}")
        return []

    return unique_docs