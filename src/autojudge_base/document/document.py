import asyncio
import itertools
import logging
from typing import Dict, Iterable, Iterator, List, Optional, Any, Set, Union
from pathlib import Path
import json


from pydantic import BaseModel, ConfigDict, PrivateAttr, ValidationError
import requests
# aiohttp, aiolimiter imported lazily in async functions

from .text_chunker import get_limit_length_chunks, get_sentence_chunks_on_newline, get_paragraph_chunks

class Document(BaseModel):
    """NeuCLIR/RAGtime documents and translations."""
    id:str
    text:str
    title:Optional[str] = None
    url:Optional[str] = None
    
    # RAGTime
    metadata:Optional[Dict[str,Any]] = None
    created:Optional[str] = None  # time stamp
    
    # old version
    cc_file:Optional[str] = None
    time:Optional[str] = None  # time stamp
    lang:Optional[str] = None
    
    model_config = ConfigDict(extra='allow')
    _full_text_chunks: Optional[List[str]] = PrivateAttr(default=None)
    _full_text_chunk_limit: Optional[int] = PrivateAttr(default=None)

    
    def get_text(self) -> str:
        """All text of the document. Add title if defined."""
        
        clean_text = self.text
        if self.title is not None:
            return self.title+" "+clean_text
        else:        
            return clean_text

    def get_document_text(self) -> str:
        return self.get_text()
    
    def get_paragraphs(self) ->List[str]:
        return get_paragraph_chunks(self.get_text())
    
    def get_sentences(self) ->List[str]:
        return get_sentence_chunks_on_newline(self.get_text())
    
    def get_text_chunks(self, limit:int) -> List[str]:
        """Break full text into chunks up to `limit`, obeying sentence boundaries."""
        if not self._full_text_chunks or limit != self._full_text_chunk_limit:
            self._full_text_chunks = get_limit_length_chunks(self.get_sentences() ,limit=limit)
            self._full_text_chunk_limit = limit
            
            if len(self._full_text_chunks)>1: 
                print(f"Breaking documents {self.id} into {len(self._full_text_chunks)} chunks of length {limit}")

        return self._full_text_chunks
    


def stream_documents_all(documents_path: Path) -> Iterator[Document]:
    with open(documents_path, 'r', encoding='utf-8') as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue  # skip blank lines

            try:
                yield Document.model_validate_json(line)
            except ValidationError as e:
                print(f"[stream_documents] Skipping line {lineno} in {documents_path}: validation error: {e}")
            except Exception as e:
                print(f"[stream_documents] Skipping line {lineno} in {documents_path}: unexpected error: {e}")

def stream_documents(
        documents_path: Path,
        wanted_doc_ids: Optional[Set[str]] = None,
    ) -> Iterator[Document]:
        with open(documents_path, 'r', encoding='utf-8') as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue  # skip blank lines

                if wanted_doc_ids and not any(doc_id in line for doc_id in wanted_doc_ids):
                    continue  # fast string-based filter

                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[stream_documents] Skipping line {lineno} in {documents_path}: invalid JSON: {e}")
                    continue

                doc_id = raw.get("id")
                if wanted_doc_ids and doc_id not in wanted_doc_ids:
                    continue  # parsed successfully, but not one of the desired ones

                try:
                    yield Document.model_validate(raw)
                except ValidationError as e:
                    print(f"[stream_documents] Skipping line {lineno} in {documents_path}: validation error: {e}")
                except Exception as e:
                    print(f"[stream_documents] Skipping line {lineno} in {documents_path}: unexpected error: {e}")

def load_documents(documents_path:Path)->List[Document]:
    documents = list()
    with open(file=documents_path, mode='r', encoding='utf-8') as f:
        for line in f.readlines():
            document = Document.model_validate_json(line) 
            documents.append(document)
            # print(document)
    return documents

#  =====

class RankedDocument(BaseModel):
    doc:Document
    rank:int
    score:Optional[float]=None

class RetrievedDocuments(BaseModel):
    query_id:str
    test_collection:str
    run_id:Optional[str]=None
    metadata:Optional[Dict[str,Any]] = None
    ranked_docs:List[RankedDocument] = list()


def load_retrieved_docs(path:Path)->List[RetrievedDocuments]:
    result = list()
    with open(file=path, mode='r', encoding='utf-8') as f:
        for line in f.readlines():
            retrieved_docs = RetrievedDocuments.model_validate_json(line) 
            result.append(retrieved_docs)
            # print(document)
    return result    


def save_retrieved_docs(path: Path, docs: List[RetrievedDocuments]) -> None:
    """Save a list of RetrievedDocuments to a JSONL file."""
    with open(file=path, mode='w', encoding='utf-8') as f:
        for doc in docs:
            # Ensure single-line JSON, no extra spaces
            json_str = doc.model_dump_json(indent=None)
            f.write(json_str + "\n")
            
#  =====


LOGGER = logging.getLogger(__name__)

# Limiter cache per event loop, keyed by (loop, max_rate)
_limiter_cache: dict = {}

def _get_limiter(max_rate: float = 3.0):
    """Get or create a limiter for the current event loop and rate."""
    from aiolimiter import AsyncLimiter
    loop = asyncio.get_running_loop()
    key = (id(loop), max_rate)
    if key not in _limiter_cache:
        _limiter_cache[key] = AsyncLimiter(max_rate=max_rate, time_period=1.0)
    return _limiter_cache[key]


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    elif isinstance(obj, set):
        return sorted(obj)
    elif isinstance(obj, Path):
        return str(obj)
    else:
        return obj

async def fetch_service_async( payload:Dict[str,str],
                            host: str,   #"10.162.95.158"
                            port: int,   # 5000,
                            command:str = "content",
                            info_str:str = "",
                            max_retries: int = 10,
                            timeout: float = 5.0,
                            rate_limit: float = 3.0,
                        ) -> Optional[Dict[str,Any]]:
            import aiohttp
            from aiohttp import ClientError

            url = f"{host}:{port}/{command}"

            json_body = json.dumps(_to_jsonable(payload), allow_nan=False)
            # print(f"url,  {url}")
            # print("payload", json_body)

            for attempt in range(1, max_retries + 1):
                async with _get_limiter(rate_limit):
                    try:
                        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                            async with session.post(url, data=json_body, headers={"Content-Type": "application/json"}) as resp:
                                if resp.status == 200:
                                                                
                                    # return await json.loads(await resp.text())   # CORRECT
                                    text_json = await resp.text()
                                    data = json.loads(text_json)
                                    return data
                                elif 500 <= resp.status < 600:
                                    LOGGER.warning(f"[{info_str}] Server error {resp.status}, retrying attempt {attempt}")
                                else:
                                    LOGGER.error(f"[{info_str}] Request failed with status {resp.status}")
                                    return None
                    except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as e:
                        LOGGER.warning(f"[{info_str}] Connection issue (attempt {attempt}): {e}")
                    except ClientError as e:
                        LOGGER.error(f"[{info_str}] Hard failure: {e}")
                        return None

                await asyncio.sleep(0.5 * attempt)  # exponential backoff

            LOGGER.error(f"[{info_str}] Exhausted retries.")
            return None


async def fetch_document_content_async(
    doc_id: str,
    collection: str, # = "ragtime-mt",
    host: str, # = "http://10.162.95.158", # ENDPOINT = "https://scale25.hltcoe.org"
    port: int, #= 5000,
    max_retries: int = 10,
    timeout: float = 5.0,
    rate_limit: float = 3.0,
) -> Optional[str]:
    if not isinstance(doc_id, (str, int)):
        raise ValueError(f"Expected a single document ID, got: {repr(doc_id)}")

    payload = {"collection": f"{collection}", "id": doc_id}

    result = await fetch_service_async(payload=payload, command="content", info_str=doc_id
                               ,host=host,port=port,max_retries=max_retries, timeout=timeout, rate_limit=rate_limit)
    if result is not None:
        return result["text"]





from typing import Iterable, Union, Dict

def fetch_document_service(
    doc_ids: Union[Iterable[str], str],
    collection_handle: str,
    parallel: bool = True,
    **kwargs
) -> Dict[str, Document]:
    """
    Fetch content for a batch of document IDs.
    Returns a dictionary: {doc_id: Document}

    Args:
        parallel: If True, fetch all documents concurrently. If False, fetch sequentially.
    """
    if isinstance(doc_ids, str):
        doc_ids = [doc_ids]

    async def fetch_all_parallel():
        tasks = [
            fetch_document_content_async(doc_id, collection=collection_handle, **kwargs)
            for doc_id in doc_ids
        ]
        texts = await asyncio.gather(*tasks)
        return {
            doc_id: Document(id=doc_id, text=text) for doc_id, text in zip(doc_ids, texts) if text is not None
        }

    async def fetch_all_sequential():
        results = {}
        for doc_id in doc_ids:
            text = await fetch_document_content_async(doc_id, collection=collection_handle, **kwargs)
            if text is not None:
                results[doc_id] = Document(id=doc_id, text=text)
        return results

    return asyncio.run(fetch_all_parallel() if parallel else fetch_all_sequential())



# def fetch_ranking_service(
#     doc_ids: Union[Iterable[str], str],
#     collection_handle: str
# ) -> Dict[str, TrecRun]:
#     """
#     Fetch content for a batch of document IDs.
#     Returns a dictionary: {doc_id: Document}
#     """
#     if isinstance(doc_ids, str):
#         doc_ids = [doc_ids]

#     async def fetch_all():
#         tasks = [
#             fetch_ranking_async(query_str = "xxx", collection=collection_handle)
#             for doc_id in doc_ids
#         ]
#         texts = await asyncio.gather(*tasks)
#         return {
#             doc_id: Document(id=doc_id, text=text) for doc_id, text in zip(doc_ids, texts) if text is not None
#         }

#     return asyncio.run(fetch_all())



def gentle_fetch_document_service(
    doc_ids: Iterable[str],
    collection_handle: str,
    batch_size:int
) -> Dict[str, Document]:
    
    def batched(iterable, n):
        it = iter(iterable)
        while (batch := list(itertools.islice(it, n))):
            yield batch

    result = dict()
    for chunk in batched(doc_ids, batch_size):
        res = fetch_document_service(doc_ids = chunk, collection_handle=collection_handle)
        result.update(res)
    return result
        

class DocMapper:
    def fetch(self, wanted_doc_ids:Set[str]) -> Dict[str,Document]:
        fetch_document_service(wanted_doc_ids)


def main():
    
    docs = fetch_document_service(doc_ids=[ \
                                            "e181ed8a-6de8-40af-a205-b46257801875_634461504"
                                            # "73e81d00-e8c2-4197-b9f0-e9542a2f16ee_139415993",
                                        #    "976e5a88-b9d0-4613-be71-b666fa22d6f3_535830456"
                                        #    "2ff26f3c-84cb-4ac7-a641-7ffa044f2470_523101866", "50f39845-d655-4e60-9928-dbb6d8b16243_721077647"
                                           ], collection_handle="ragtime-mt")
    
    for doc in docs:
        # print(docs[doc].id, docs[doc].text)
        # print()
        # print(docs[doc].get_document_text())
        # print(get_sentence_chunks_blingfire(docs[doc].get_text()))
        print(get_limit_length_chunks(docs[doc].get_sentences()[0], limit=100))
        # print(docs[doc].get_text_chunks(limit=100))


if __name__ == "__main__":
    main()
