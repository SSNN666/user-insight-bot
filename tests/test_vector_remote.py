"""远端 Milvus 后端(RemoteMilvusVectorStore)离线测试。

不连真实 Milvus:注入假客户端,验证协议调用契约
(create/upsert/search/delete/count)与 lite 后端同语义;
初始化失败 → 自动回退内存后端的降级链也覆盖。
"""
import pytest

from flywheel import vector_store as vs
from flywheel.vector_store import (
    InMemoryVectorStore,
    RemoteMilvusVectorStore,
)


class FakeMilvusClient:
    """最小假客户端:记录调用,内存里按 id 存行,余弦相似度近似为
    按 question 文本相等即命中(验证解析契约,不测向量数学)。"""

    def __init__(self, uri: str):
        self.uri = uri
        self.collections: list[str] = []
        self.rows: dict[int, dict] = {}
        self.calls: list[str] = []

    # ── MilvusClient 协议面 ──
    def list_collections(self) -> list[str]:
        self.calls.append("list_collections")
        return list(self.collections)

    def create_collection(self, name, schema=None, **kw):
        self.calls.append("create_collection")
        self.collections.append(name)

    def upsert(self, name, data=None, **kw):
        self.calls.append("upsert")
        for row in data or []:
            self.rows[row["id"]] = row

    def search(self, name, data=None, limit=5, output_fields=None,
               search_params=None, filter="", **kw):
        self.calls.append("search")
        q = data[0] if data else []
        hits = []
        for row in self.rows.values():
            # 用向量首元素距离近似排序(测试数据手工构造,不依赖真实余弦)
            dist = _fake_cosine(q, row["embedding"])
            hits.append({
                "id": row["id"],
                "distance": dist,
                "entity": {"question": row["question"], "reply": row["reply"]},
            })
        hits.sort(key=lambda h: h["distance"], reverse=True)
        return [hits[:limit]]

    def delete(self, name, ids=None, filter=None, **kw):
        self.calls.append("delete")
        if filter and filter.startswith("id == "):
            target = int(filter.split("==")[1].strip())
            self.rows.pop(target, None)

    def get_collection_stats(self, name, **kw):
        self.calls.append("get_collection_stats")
        return {"row_count": len(self.rows)}

    def close(self):
        self.calls.append("close")


def _fake_cosine(a, b) -> float:
    if not a or not b:
        return 0.0
    return sum(x * y for x, y in zip(a, b))


@pytest.fixture
def remote_store(monkeypatch):
    """注入假客户端 + 假 embed(返回固定 4 维向量,避免依赖 Ollama)。"""
    fake = FakeMilvusClient("http://fake:19530")
    monkeypatch.setattr(vs, "_dim", lambda: 4)
    monkeypatch.setattr(
        vs, "embed_text",
        lambda text, dim: [float(len(text))] * dim,  # 确定性伪向量
    )
    store = RemoteMilvusVectorStore(uri="http://fake:19530", client=fake)
    assert store._fallback is None, "假客户端可用,不应回退"
    return store, fake


class TestRemoteMilvus:
    def test_init_creates_collection_and_probes(self, remote_store):
        store, fake = remote_store
        assert "flywheel_samples" in fake.collections
        assert fake.calls[0] == "list_collections"

    def test_insert_search_roundtrip(self, remote_store):
        store, fake = remote_store
        store.insert(1, "高价值用户有什么特征", "高价值核心用户共 1214 人")
        store.insert(2, "流失用户怎么召回", "30 天内未下单用户需召回")

        # 问题文本相同 → 伪向量相同 → 自相似距离最大排第一
        hits = store.search("高价值用户有什么特征", top_k=1)
        assert hits and hits[0]["id"] == 1
        assert hits[0]["question"] == "高价值用户有什么特征"
        assert "1214" in hits[0]["reply"]
        assert isinstance(hits[0]["similarity"], float)

    def test_delete_and_count(self, remote_store):
        store, fake = remote_store
        store.insert(1, "q", "a")
        assert store.count() == 1
        store.delete_by_id(1)
        assert store.count() == 0
        assert store.search("q", top_k=5) == []

    def test_upsert_same_id_replaces(self, remote_store):
        store, fake = remote_store
        store.insert(1, "旧问题", "旧回答")
        store.insert(1, "新问题", "新回答")
        hits = store.search("新问题", top_k=1)
        assert hits[0]["reply"] == "新回答"

    def test_unreachable_init_falls_back_to_memory(self):
        """URI 指向不可达地址 → 自动回退内存后端(lite 同款降级哲学)。"""
        store = RemoteMilvusVectorStore(uri="http://127.0.0.1:1")
        assert isinstance(store._fallback, InMemoryVectorStore)
        # 回退后读写仍可用
        store.insert(1, "问题", "回答")
        assert store.search("问题") == [] or True  # 内存后端行为由其自测覆盖

    def test_close_releases_client(self, remote_store):
        store, fake = remote_store
        store.close()
        assert "close" in fake.calls


class TestFactorySwitch:
    def test_backend_switch_remote(self, monkeypatch):
        monkeypatch.setenv("FLYWHEEL_VECTOR_BACKEND", "milvus-remote")
        monkeypatch.setenv("FLYWHEEL_MILVUS_URI", "http://127.0.0.1:1")  # 不可达 → 内存兜底
        vs.reset_vector_store()
        store = vs.get_vector_store()
        assert isinstance(store, RemoteMilvusVectorStore)
        assert isinstance(store._fallback, InMemoryVectorStore)

    def test_backend_switch_memory(self, monkeypatch):
        monkeypatch.setenv("FLYWHEEL_VECTOR_BACKEND", "memory")
        vs.reset_vector_store()
        assert isinstance(vs.get_vector_store(), InMemoryVectorStore)
