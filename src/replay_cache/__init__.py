from typing import Optional, Any

from langchain_core.caches import BaseCache, RETURN_VAL_TYPE
from langchain_core.globals import set_llm_cache, get_llm_cache
from contextlib import contextmanager
from collections import Counter


@contextmanager
def replay_cache(cache: BaseCache):
    """
    `replay_cache` wraps any cache from LangChain and remembers not only prompt & LLM configuration but also
    adds a sequential number to the cached value.
    So when it is used for the first time,
    it will only fill the cache and query the model each time.

    When `replay_cache` is used again over the same cache, then it will return
    the results from the cache in the order
    how it was obtained in the previous run (on the same prompts over the same models).
    Each pair prompt & LLM configuration remembers its own order
    of responses. If a query is called more times than in the previous run,
    then a model is queried again and the result is stored in cached.

    # Example

    ```python
    from replay_cache import replay_cache

    cache = InMemoryCache()

    with replay_cache(cache):
        llm.invoke("How are you?")  # Answer A (model queried)
        llm.invoke("How are you?")  # Answer B (model queried)
        llm.invoke("How are you?")  # Answer C (model queried)

    with replay_cache(cache):
        llm.invoke("How are you?")  # Answer A (taken from cache)
        llm.invoke("How are you?")  # Answer B (taken from cache)
        llm.invoke("How are you?")  # Answer C (taken from cache)
        llm.invoke("How are you?")  # Answer D (model queried)

    with replay_cache(cache):
        llm.invoke("How are you?")  # Answer A (taken from cache)
        llm.invoke("How are you?")  # Answer B (taken from cache)
    ```

    ```python
    from replay_cache import replay_cache

    cache = InMemoryCache()

    with replay_cache(cache):
        llm.invoke("How are you?")        # Answer A (model queried)
        llm.invoke("How are you?")        # Answer B (model queried)
        llm.invoke("What is your name?")  # Answer C (model queried)
        llm.invoke("What is your name?")  # Answer D (model queried)

    with replay_cache(cache):
        llm.invoke("How are you?")        # Answer A (taken from cache)
        llm.invoke("What is your name?")  # Answer C (taken from cache)
        llm.invoke("How are you?")        # Answer B (taken from cache)
        llm.invoke("What is your name?")  # Answer D (taken from cache)
    ```
    """

    old_cache = get_llm_cache()
    rcache = ReplayCache(cache)
    set_llm_cache(rcache)
    try:
        yield rcache
    finally:
        set_llm_cache(old_cache)


class ReplayCache(BaseCache):
    def __init__(self, cache: BaseCache) -> None:
        self._cache = cache
        self._counters = Counter()

    def _get_inner_llm_string(self, key):
        return f"{key[1]}_replay_{self._counters[key]}"

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        key = (prompt, llm_string)
        result = self._cache.lookup(prompt, self._get_inner_llm_string(key))
        if result is not None:
            self._counters[key] += 1
        return result

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        key = (prompt, llm_string)
        self._cache.update(prompt, self._get_inner_llm_string(key), return_val)
        self._counters[key] += 1

    def clear(self, **kwargs: Any) -> None:
        self._counters = Counter()
        self._cache.clear()
