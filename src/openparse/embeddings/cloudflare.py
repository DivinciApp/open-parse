import time
import logging
from typing import List, Literal
import requests
from requests.exceptions import RequestException, SSLError
import backoff
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# Logger setup
cf_logger = logging.getLogger('cloudflare')
cf_logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - 🟠☁️ CLOUDFLARE - %(message)s')
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
        batch_size: int = 100,
        max_retries: int = 3,
        retry_delay: int = 2,
        **kwargs
    ):
        api_token = kwargs.get('api_token', None)
        account_id = kwargs.get('account_id', None)
        if not api_token:
            raise ValueError("❌ Cloudflare API token (api_token) required.")
        if not account_id:
            raise ValueError("❌ Cloudflare Account ID (account_id) required.")

        # Set instance variables first
        self.model = model
        self.api_token = api_token
        self.account_id = account_id
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.base_url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run"
        
        # Create session after setting variables
        self.session = self._create_session()
        
        # Test connection
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
        (RequestException, SSLError),
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
            json = response.json()
            
            if json.get('success') is not True:
                raise ValueError(f"❌ Error from Cloudflare API: {json.get('errors')}")

            result = json.get('result')
            if not result:
                raise ValueError(f"❌ Unexpected response format: {json}")
            dataList = result.get('data')
            if not dataList:
                raise ValueError(f"❌ Unexpected response format: {json}")
            data = dataList[0]
            if not data:
                raise ValueError(f"❌ Unexpected response format: {json}")

            return data
            
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