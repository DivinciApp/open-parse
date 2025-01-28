import os
import logging
from typing import List, Literal, Optional
import requests
import time
import backoff
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from requests.exceptions import RequestException

# Logger setup
cf_logger = logging.getLogger('cloudflare')
cf_logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - 🟠☁️CLOUDFLARE - %(message)s')
handler.setFormatter(formatter)
cf_logger.addHandler(handler)

CloudflareModel = Literal[
    "@cf/baai/bge-small-en-v1.5",
    "@cf/baai/bge-base-en-v1.5",
    "@cf/baai/bge-large-en-v1.5",
]

class CloudflareEmbeddings:
    def __init__(
        self,
        model: CloudflareModel = "@cf/baai/bge-base-en-v1.5",
        api_token: Optional[str] = None,
        account_id: Optional[str] = None,
        batch_size: int = 100,
        max_retries: int = 3,
        retry_delay: int = 2
    ):
        self.model = model
        self.session = self._create_session()
        self.api_token = api_token or os.environ.get("CLOUDFLARE_API_TOKEN")
        self.account_id = account_id or os.environ.get("CLOUDFLARE_ACCOUNT_ID")
        self.batch_size = min(batch_size, 100)  # CF max batch size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        if not self.api_token or not self.account_id:
            raise ValueError("❌ Cloudflare API token and account ID required.")
            
        self.base_url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run"
        self._check_connection()
        
    def _check_connection(self) -> None:
        """Test connection to Cloudflare API"""
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.get(
                    f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}",
                    headers=headers
                )
                response.raise_for_status()
                return
            except RequestException as e:
                if attempt == self.max_retries - 1:
                    raise ConnectionError(f"❌ Failed to connect to Cloudflare API: {str(e)}")
                time.sleep(self.retry_delay)

    def _create_session(self):
        session = requests.Session()
        retry_strategy = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=10
        )
        session.mount("https://", adapter)
        session.headers.update({
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        })
        return session

    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException, SSLError),
        max_tries=5
    )

    def _get_embedding(self, text: str) -> List[float]:
        try:
            text_preview = text[:50] + "..." if len(text) > 50 else text
            cf_logger.info(f"Embedding text: {text_preview}")
            
            response = self.session.post(
                f"{self.base_url}/{self.model}",
                json={"text": text},
                timeout=30,
                verify=True
            )
            
            if response.status_code != 200:
                cf_logger.error(f"Error response: {response.text}")
            
            response.raise_for_status()
            result = response.json()
            
            if not result.get('data') or not result['data'][0]:
                raise ValueError(f"Unexpected response format: {result}")
                
            return result['data'][0]
            
        except Exception as e:
            cf_logger.error(f"Failed: {str(e)}")
            raise

    def embed_many(self, texts: List[str]) -> List[List[float]]:
        res = []
        non_empty_texts = [text for text in texts if text]

        for i in range(0, len(non_empty_texts), self.batch_size):
            batch_texts = non_empty_texts[i:i + self.batch_size]
            batch_embeddings = [self._get_embedding(text) for text in batch_texts]
            res.extend(batch_embeddings)

        # Pad empty texts with zeros
        embedding_size = len(res[0]) if res else 768  # BGE base size
        return res + [[0.0] * embedding_size] * (len(texts) - len(non_empty_texts))