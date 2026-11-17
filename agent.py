"""Manufacturing assistant agent.

The LLM decides when to search the manuals and when to run the scheduler.
Loop: send messages + tool schemas -> if the model returns tool_calls, run them
and append the results as tool messages -> repeat until it answers in plain text.
Each turn (question, tool calls, latency, tokens) is appended to data/agent_log.jsonl.

Usage: python agent.py
"""
import json
import time
from pathlib import Path

from config import CHAT_MODEL, chat_client
from optimizer import load_shop, optimize_schedule
from rag import format_context, get_index

BASE = Path(__file__).parent
LOG_FILE = BASE / "data" / "agent_log.jsonl"
MAX_STEPS = 6  # stop runaway tool loops

SYSTEM_PROMPT = """你是制造车间的智能助手，可以使用三个工具：
- search_manual：查设备手册（加工中心 HR-850、AGV T300、注塑机 IM-250）。凡是设备操作、报警、保养、故障的问题都要先查手册，不要凭记忆回答
- get_shop_info：查看车间有哪些设备和待加工任务
- optimize_schedule：用 NSGA-II 做排产优化，同时考虑完工时间和能耗

回答要求：
1. 用手册回答时，只能写 search_manual 返回的资料里有的内容，每条结论后面标出来源，例如（cnc_hr850.md · 6. 报警代码表）；
   资料里没有的步骤、数值、注意事项一律不要写。手册里查不到就直说"手册中没有找到"。
   如果确实需要补充通用经验，单独放在最后，标题写"手册外的一般建议（未经手册确认）"
2. 排产结果要说明：推荐方案的完工时间和能耗、它在 Pareto 前沿上的取舍（和工具返回的 fastest、lowest_energy 两个方案比较），以及每台设备分到哪些任务。
   数字只能用工具返回的，不要自己推算前沿上的最快或最省电方案
3. 用户说某台设备坏了、停机、在维修，就把它放进 unavailable_machines 重新优化，并和之前的结果对比
4. 用中文，简洁，分点"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_manual",
            "description": "在设备手册中检索相关段落，用于回答设备操作、报警代码、保养、故障排除等问题",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索用的问题或关键词，例如'ALM 2010 气压不足怎么处理'"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_shop_info",
            "description": "查看车间的设备列表（编号、类型、速度、功率）和待加工任务",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "optimize_schedule",
            "description": "用 NSGA-II 把待加工任务分配到设备上，同时最小化完工时间和总能耗，返回 Pareto 前沿和推荐方案",
            "parameters": {
                "type": "object",
                "properties": {
                    "unavailable_machines": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "停机、故障或维修中的设备编号，例如 ['CNC-2']",
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["balanced", "time", "energy"],
                        "description": "balanced=兼顾两者（默认），time=尽快完工，energy=尽量省电",
                    },
                },
            },
        },
    },
]


def search_manual(query):
    hits = get_index().search(query, k=4)
    return format_context(hits)


def get_shop_info():
    shop = load_shop()
    return {"machines": shop["machines"], "task_count": len(shop["tasks"]), "tasks": shop["tasks"]}


TOOL_FUNCS = {
    "search_manual": search_manual,
    "get_shop_info": get_shop_info,
    "optimize_schedule": optimize_schedule,
}


def run_tool(name, args):
    """Errors go back to the model as tool output instead of crashing the loop."""
    try:
        result = TOOL_FUNCS[name](**args)
    except Exception as e:
        result = {"error": f"{type(e).__name__}: {e}"}
    return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)


def chat(messages, verbose=True):
    """Run one agent turn, appending to messages in place. Returns (answer, log)."""
    log = {"question": messages[-1]["content"], "tool_calls": [], "tokens": 0}
    start = time.time()
    for _ in range(MAX_STEPS):
        resp = chat_client.chat.completions.create(
            model=CHAT_MODEL, messages=messages, tools=TOOLS, temperature=0.2,
        )
        msg = resp.choices[0].message
        if resp.usage:
            log["tokens"] += resp.usage.total_tokens

        if not msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content})
            break

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
        })
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            if verbose:
                print(f"  [tool] {tc.function.name}({json.dumps(args, ensure_ascii=False)})")
            log["tool_calls"].append({"name": tc.function.name, "args": args})
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": run_tool(tc.function.name, args),
            })
    else:
        messages.append({"role": "assistant", "content": "（工具调用次数过多，已停止，请换个问法再试。）"})

    log["seconds"] = round(time.time() - start, 2)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(log, ensure_ascii=False) + "\n")
    return messages[-1]["content"], log


def main():
    print("Manufacturing assistant (q to quit, /new to reset)")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    while True:
        question = input("\nYou: ").strip()
        if question.lower() == "q":
            break
        if question == "/new":
            messages = messages[:1]
            print("(conversation reset)")
            continue
        if not question:
            continue
        n_before = len(messages)
        messages.append({"role": "user", "content": question})
        try:
            text, _ = chat(messages)
        except Exception as e:  # roll back this turn so the history stays valid
            del messages[n_before:]
            print(f"Error: {e}")
            continue
        print(f"\nAssistant: {text}")


if __name__ == "__main__":
    main()
