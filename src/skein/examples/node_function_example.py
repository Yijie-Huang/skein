"""Example demonstrating NodeFunction usage."""

from __future__ import annotations

import asyncio
from skein.core import Graph, BaseState, Node
from skein.exporters import InMemoryExporter


# 方式1：定义函数 node（使用 NodeFunction 类型）
async def fetch_data(state: BaseState) -> BaseState:
    """Fetch data from source."""
    print(f"[fetch_data] Processing trace {state.trace_id}")
    # 模拟异步操作
    await asyncio.sleep(0.1)
    # 可以添加自定义属性（如果 state 是子类）
    state.current_node = "fetch_data"
    return state


async def process_data(state: BaseState) -> BaseState:
    """Process the fetched data."""
    # Graph 在执行节点前就把 current_node 设为当前节点名，这里读到的是 "process" 而非上一个节点
    print(f"[process_data] Trace {state.trace_id}, current node: {state.current_node}")
    await asyncio.sleep(0.1)
    state.current_node = "process_data"
    return state


# 方式2：定义 Node 类（用于复杂逻辑）
class ValidateNode(Node):
    """Validate processed data."""

    def __init__(self):
        super().__init__("validate")

    async def run(self, state: BaseState) -> BaseState:
        print(f"[validate] Checking state, current node: {state.current_node}")
        await asyncio.sleep(0.1)
        state.current_node = "validate"
        return state


async def main():
    # 创建图
    graph = Graph("node_function_example", InMemoryExporter())

    # 两种 node 可以混用：函数 node 与 Node 子类
    graph.add_node("fetch", fetch_data)
    graph.add_node("process", process_data)
    graph.add_node("validate", ValidateNode())

    # 定义执行流
    graph.add_edge("fetch", "process")
    graph.add_edge("process", "validate")
    graph.set_entry_point("fetch")

    # 创建初始状态
    state = BaseState(trace_id="example-001")

    # 执行
    print("Starting workflow...")
    result = await graph.run(state)
    print(f"Completed! Status: {result.status}")
    print(f"Final node: {result.current_node}")

    print("Execution trace:")
    for event in graph.exporter.events:
        print(f"  {event.node_name}: {event.status}")


if __name__ == "__main__":
    asyncio.run(main())
