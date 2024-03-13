# Replaying cache for LangChain

This is a replay cache implementation for LangChain.
It is useful when you want to make the same LLM queries, and you do not want to get
a single answer from the cache each time.

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

## Installation

```commandline
$ pip install replay_cache
```

## Example

### Without replay cache

```python
cache = InMemoryCache()
set_llm_cache(cache)

llm.invoke("How are you?")  # Answer A (model queried)
llm.invoke("How are you?")  # Answer A (taken from cache)
llm.invoke("How are you?")  # Answer A (taken from cache)
```

### With replay cache

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
