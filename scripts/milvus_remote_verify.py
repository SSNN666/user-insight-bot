"""② 向量后端 milvus-remote 端到端验证(真实 Milvus standalone)。

前置:
  1) docker compose -f docker-compose.milvus.yml up -d   # etcd+MinIO+Milvus
  2) 本脚本在 FLYWHEEL_VECTOR_BACKEND=milvus-remote 下做真实读写往返:
     insert → count → search(同文本自相似第一) → delete → count=0,
     全部走 pymilvus gRPC 连接独立 Milvus,embedding 用确定性伪向量
     (不依赖 Ollama,隔离向量数学,只验证远端存储协议)。

用法: uv run python scripts/milvus_remote_verify.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("FLYWHEEL_VECTOR_BACKEND", "milvus-remote")
os.environ.setdefault("FLYWHEEL_MILVUS_URI", "http://127.0.0.1:19530")

from config.settings import get_settings
get_settings.cache_clear()

from flywheel import vector_store as vs


def main() -> None:
    # 注:RemoteMilvusVectorStore 初始化时会探测集合;embed 伪向量与
    # 集合 dim(768) 对齐 —— 直接 monkeypatch 模块级 embed_text。
    calls: dict = {}

    def fake_embed(text: str, dim: int):
        calls.setdefault("embeds", 0)
        calls["embeds"] += 1
        # 按文本确定性向量(同文本同向量、异文本异向量——之前用常量向量导致
        # 删一条后另一条同向量仍命中,误判为墓碑不可见)
        import hashlib
        seed = int.from_bytes(hashlib.md5(text.encode()).digest()[:8], "little")
        rng = __import__("random").Random(seed)
        return [rng.random() for _ in range(dim)]

    vs.embed_text = fake_embed
    vs.reset_vector_store()

    store = vs.get_vector_store()
    assert store.__class__.__name__ == "RemoteMilvusVectorStore", \
        f"后端不是 milvus-remote: {store.__class__.__name__}"
    assert store._fallback is None, "远端不可达?看上面日志(应自动回退前会打 warning)"

    store.insert(1, "高价值用户有什么特征", "高价值核心用户共 1214 人")
    store.insert(2, "流失用户怎么召回", "30 天内未下单用户需召回")
    print(f"[insert] 2 条写入,count={store.count()}")

    hits = store.search("高价值用户有什么特征", top_k=1)
    assert hits and hits[0]["id"] == 1, hits
    assert "1214" in hits[0]["reply"], hits
    print(f"[search] 自相似命中 id=1, similarity={hits[0]['similarity']}")

    store.delete_by_id(1)
    assert store.count() == 1, store.count()
    # 删除语义:被删行不再出现在检索结果(id 1 消失);ANN 邻居(id 2)仍在
    # 属正常(cosine 随机向量间也有相似度)——断言"结果为空"是错误语义
    hits = store.search("高价值用户有什么特征", top_k=5)
    ids = [h["id"] for h in hits]
    assert 1 not in ids, f"删除后 id=1 仍可见: {ids}"
    assert 2 in ids, f"邻居 id=2 应仍在: {ids}"
    print(f"[delete] id=1 已删且检索不可见,id=2 邻居仍在: {ids}")

    store.delete_by_id(2)
    assert store.count() == 0
    print(f"[cleanup] 清理完成,count=0;embedding 调用 {calls.get('embeds', 0)} 次")
    print("ALL PASS - milvus-remote real Milvus E2E")


if __name__ == "__main__":
    main()
