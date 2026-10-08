import sys
from pathlib import Path

# 获取当前脚本所在文件夹
current_file_dir = Path(__file__).parent
# 向上跳一层，得到src目录
src_root = current_file_dir.parent
sys.path.append(str(src_root))

from rag_hybrid_query import hybrid_query
from logger import logger



# 测试用例库，包含最终验收案例
TEST_QUESTIONS = [
    "广州市长隆旅游度假区在哪里",
    "查询广州长隆度假区开放时间",   # 【保留的最终验收测试点】
    "肇庆有哪些5A级景区",
    "丹霞山门票多少钱"
]

def batch_test():
    logger.info("=======开始批量RAG功能测试=======")
    for q in TEST_QUESTIONS:
        logger.info(f"测试问题：{q}")
        try:
            result = hybrid_query(q)
            logger.info(f"生成回答：{result}")
        except Exception as e:
            logger.error(f"问题【{q}】执行失败，异常信息:{str(e)}",exc_info=True)
    logger.info("=======批量测试全部完成=======")

if __name__ == "__main__":
    batch_test()
