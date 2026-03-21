"""
测试知识图谱实体检索效果对比
- 方案A: 传统字符串匹配（_get_relevant_context_by_keywords，fallback方案）
- 方案B: embedding 向量检索（get_relevant_context，新默认方案）

用法：先确保 kb_7 有数据，运行此脚本测试
"""

import asyncio
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "backend"))

from app.core.config import settings
from app.services.llm import get_embedding_provider
from app.services.knowledge_graph_service import KnowledgeGraphService


class TestEntitySearch:
    """测试实体检索效果"""

    def __init__(self):
        self.emb_provider = None
        self.rag = None

    async def initialize(self):
        """初始化 KnowledgeGraphService 实例"""
        print("Initializing KnowledgeGraphService...")

        # 获取 embedding provider（用于方案B的向量计算）
        self.emb_provider = get_embedding_provider()

        # 创建 KG service（workspace 7）
        self.rag = KnowledgeGraphService(workspace_id=7)
        await self.rag._get_rag()  # 预初始化

        print(f"[OK] KnowledgeGraphService initialized")
        print(f"  Working dir: {self.rag.working_dir}")
        print(f"  Embedding dim: {self.emb_provider.get_dimension()}")

    async def close(self):
        """清理资源"""
        if self.rag:
            await self.rag.cleanup()

    # ========== 方案A：fallback关键词匹配 ==========
    async def search_by_keywords(self, question: str, max_entities: int = 20) -> list[dict]:
        """
        方案A: 传统关键词匹配（fallback逻辑）
        直接调用 KnowledgeGraphService 的内部 fallback 方法
        """
        # 需要先获取原始数据，因为 fallback 方法需要传入
        rag = self.rag
        storage = rag._rag.chunk_entity_relation_graph

        try:
            all_nodes = await storage.get_all_nodes()
            all_edges = await storage.get_all_edges()
        except Exception as e:
            print(f"  [ERROR] Get nodes failed: {e}")
            return []

        # 调用 fallback 方法
        result_str = await self.rag._get_relevant_context_by_keywords(
            question, max_entities, 30, all_nodes, all_edges
        )

        # 解析 result_str 返回实体列表
        if not result_str:
            return []

        entities = []
        lines = result_str.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("- "):
                # 格式: "- EntityName [Type]: description"
                line = line[2:]  # 移除 "- "
                if ":" in line:
                    name_type, desc = line.split(":", 1)
                    name = name_type.strip()
                    # 解析类型（如果有）
                    if "[" in name and "]" in name:
                        bracket_start = name.rfind("[")
                        bracket_end = name.rfind("]")
                        entity_type = name[bracket_start+1:bracket_end]
                        entity_name = name[:bracket_start].strip()
                    else:
                        entity_type = "Unknown"
                        entity_name = name
                    entities.append({
                        "entity_name": entity_name,
                        "entity_type": entity_type,
                        "description": desc.strip()
                    })
                else:
                    entities.append({
                        "entity_name": line,
                        "entity_type": "Unknown",
                        "description": ""
                    })

        return entities

    # ========== 方案B：Embedding向量检索 ==========
    async def search_by_embedding(self, question: str, max_entities: int = 20) -> list[dict]:
        """
        方案B: 使用 embedding 在 entities_vdb 中搜索（新的默认方法）
        """
        rag = self.rag
        storage = rag._rag.chunk_entity_relation_graph

        # 1. 获取所有节点和边（用于后续关系查找）
        try:
            all_nodes = await storage.get_all_nodes()
            all_edges = await storage.get_all_edges()
        except Exception as e:
            print(f"  [ERROR] Get nodes failed: {e}")
            return []

        # 创建节点查找字典
        node_dict = {n.get("id", ""): n for n in all_nodes}

        # 2. 计算 query embedding
        try:
            query_embedding = await self.emb_provider.embed([question])
            query_embedding = query_embedding[0]
        except Exception as e:
            print(f"  [ERROR] Embedding failed: {e}")
            return []

        # 3. 在 entities_vdb 中向量搜索
        entities_vdb = rag._rag.entities_vdb
        results = await entities_vdb.query(
            query=question,
            top_k=max_entities,
            query_embedding=query_embedding
        )

        # 4. 格式化结果 + 补充关系信息
        matched_entities = []
        matched_entity_names = []

        for r in results:
            # LightRAG 存储的 content 格式: "entity_name\ndescription"
            content = r.get("content", "")
            entity_name = r.get("entity_name", "")

            if content and "\n" in content:
                parts = content.split("\n", 1)
                name = parts[0]
                description = parts[1] if len(parts) > 1 else ""
            elif entity_name:
                name = entity_name
                description = r.get("description", "")
            else:
                continue

            matched_entity_names.append(name)

            # 获取 entity_type（从原始节点数据中查找）
            entity_type = "Unknown"
            if name in node_dict:
                entity_type = node_dict[name].get("entity_type", "Unknown")
                description = description or node_dict[name].get("description", "")

            matched_entities.append({
                "entity_name": name,
                "entity_type": entity_type,
                "description": description
            })

        # 5. 查找相关关系（可选：补充与这些实体相关的边）
        # 暂时不在此搜索，因为方案A和方案C会做

        return matched_entities

    # ========== 对比测试 ==========
    async def compare_search(self, queries: list[str]):
        """对比方案A和方案B两种检索方法"""
        print("\n" + "="*80)
        print("Starting entity search comparison")
        print("="*80)

        for query in queries:
            print(f"\n{'='*80}")
            print(f"Query: {query}")
            print(f"{'='*80}")

            # Scheme A: keyword matching (fallback)
            print("\n[Scheme A] Keyword matching (fallback):")
            result_a = await self.search_by_keywords(query)
            print(f"  Matched {len(result_a)} entities:")
            for i, e in enumerate(result_a[:10], 1):
                name = e['entity_name']
                etype = e['entity_type']
                desc = e['description'][:60] if e['description'] else ""
                print(f"    {i}. {name} [{etype}]")
                if desc:
                    print(f"       {desc}...")

            # Scheme B: embedding search
            print("\n[Scheme B] Embedding vector search:")
            result_b = await self.search_by_embedding(query)
            print(f"  Matched {len(result_b)} entities:")
            for i, e in enumerate(result_b[:10], 1):
                name = e['entity_name']
                etype = e['entity_type']
                desc = e['description'][:60] if e['description'] else ""
                print(f"    {i}. {name} [{etype}]")
                if desc:
                    print(f"       {desc}...")

            # Comparison analysis
            print("\n[Comparison]")
            names_a = {e['entity_name'] for e in result_a}
            names_b = {e['entity_name'] for e in result_b}
            overlap = names_a & names_b
            a_only = names_a - names_b
            b_only = names_b - names_a
            print(f"  A only: {len(a_only)} entities")
            print(f"  B only: {len(b_only)} entities")
            print(f"  Overlap: {len(overlap)} entities")
            if overlap:
                print(f"    Common: {', '.join(list(overlap)[:5])}")
            if a_only:
                print(f"    A-only (keyword match):")
                for name in list(a_only)[:3]:
                    print(f"      - {name}")
            if b_only:
                print(f"    B-only (semantic match):")
                for name in list(b_only)[:3]:
                    print(f"      - {name}")


async def main():
    """Main: define test queries and run comparison"""
    test_queries = [
        # English queries (basic)
        "What is MoE-Mamba?",
        "NeurIPS 2021",

        # Chinese queries (cross-language test) - KEY!
        "MoE架构",  # Should match "Mixture of Experts", "MoE-Mamba", "ST-MoE"
        "动作预测模型有哪些？",  # Should match models like MoE-Mamba, ST-MoE, TBIFormer
        "Mamba模型",  # Should match "Mamba", "Bi-TMamba", "Bi-SMamba"
    ]

    tester = TestEntitySearch()
    try:
        await tester.initialize()

        print("\n" + "="*80)
        print("Running entity search comparison (A: keyword vs B: embedding)...")
        print("="*80)
        await tester.compare_search(test_queries)

    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await tester.close()


if __name__ == "__main__":
    asyncio.run(main())
