from langchain_community.cache import InMemoryCache
from langchain_community.llms.fake import FakeListLLM

from replay_cache import replay_cache


def test_replay_cache():
    llm = FakeListLLM(responses=["One", "Two", "Three", "Four", "Five", "Six", "Seven"])
    cache = InMemoryCache()

    with replay_cache(cache):
        assert llm.invoke("Hello!") == "One"
        assert llm.invoke("Hello!") == "Two"
        assert llm.invoke("Bye!") == "Three"

    with replay_cache(cache):
        assert llm.invoke("Bye!") == "Three"
        assert llm.invoke("Hello!") == "One"
        assert llm.invoke("Hello!") == "Two"

    with replay_cache(cache):
        assert llm.invoke("Hello!") == "One"
        assert llm.invoke("Hello!") == "Two"
        assert llm.invoke("Bye!") == "Three"
        assert llm.invoke("Hello!") == "Four"
        assert llm.invoke("Bye!") == "Five"

    with replay_cache(cache):
        assert llm.invoke("Ok?") == "Six"

    with replay_cache(cache):
        assert llm.invoke("Bye!") == "Three"
        assert llm.invoke("Bye!") == "Five"
        assert llm.invoke("Bye!") == "Seven"
        assert llm.invoke("Hello!") == "One"
        assert llm.invoke("Hello!") == "Two"
        assert llm.invoke("Hello!") == "Four"
        assert llm.invoke("Ok?") == "Six"
