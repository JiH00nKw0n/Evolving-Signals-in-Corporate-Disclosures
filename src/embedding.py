from typing import List, Dict

import torch
from tqdm import tqdm

from src.client import client


def _get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = _get_device()


@torch.no_grad()
async def get_embedding(texts: List[str], model="text-embedding-3-large", print_usage: bool = False):
    texts = [text.replace("\n", " ") for text in texts]
    response = await client.embeddings.create(input=texts, model=model)
    if print_usage:
        print(response.usage)
    embeddings = [torch.tensor(data.embedding).to(DEVICE) for data in response.data]

    return torch.stack(embeddings)

@torch.no_grad()
async def get_embedding_with_cache(texts: List[str], state_dict: Dict[str, torch.Tensor],
                                   model="text-embedding-3-large", print_usage: bool = False, return_list: bool = False):
    uncached_texts = []
    text_to_index = {}

    for i, text in tqdm(enumerate(texts), total=len(texts)):
        if text not in state_dict:
            uncached_texts.append(text)
        text_to_index[text] = i

    if uncached_texts:
        print(f"Getting embeddings for {len(uncached_texts)} uncached texts using API (cached: {len(texts) - len(uncached_texts)})")
        new_embeddings = await get_embedding(uncached_texts, model, print_usage)
        for i, text in enumerate(uncached_texts):
            state_dict[text] = new_embeddings[i]
    else:
        print(f"All {len(texts)} texts found in cache")

    if texts:
        print("Building result embeddings from cache...")
        result_embeddings = [state_dict[text] for text in tqdm(texts, desc="Loading embeddings")]
        if return_list:
            return result_embeddings
        return torch.stack(result_embeddings)
    else:
        return []

@torch.no_grad()
def get_similarity(a: torch.Tensor, b: torch.Tensor, is_normalized: bool = True):
    if not is_normalized:
        a = torch.nn.functional.normalize(a, p=2, dim=-1)
        b = torch.nn.functional.normalize(b, p=2, dim=-1)
    return torch.mm(a, b.T)
