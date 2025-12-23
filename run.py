import time

import torch

from model import model_class
import tools
from tools import search_tool, git_tool

# chat_bot = model_class.ChatBot()
# chat_bot.model.eval()
# torch.cuda.empty_cache()
# with torch.no_grad():
#     while True:
#         user_input = input("입력해주세요: ")
#         if user_input.lower().strip()=="exit":
#             break
#         print(f"입력:\n{user_input}\n")
#         print(f"히스토리:\n{chat_bot.history}\n")
#         gen_start_time = time.time()
#         print(f"답변:\n{chat_bot.generate(user_input)}\n")
#         gen_end_time = time.time()
#         print(f"\n생성 시간: {gen_end_time - gen_start_time:.2f} 초\n")

#         print("\n-------------------------------------------------------\n")

total_tools: dict[str, tools.Tool] = {
    search_tool.WebSearchTool.name: search_tool.WebSearchTool()
}
agent = model_class.Agent(total_tools=total_tools)

torch.cuda.empty_cache()
with torch.no_grad():
    while True:
        user_input = input("입력: ")
        user_input = user_input.lower().strip()
        if user_input == "exit":
            break
        print(f"\n입력:\n{user_input}\n")

        gen_start_time = time.time()
        print(f"답변:\n{agent.run(user_input)}\n")
        gen_end_time = time.time()
        print(f"\n생성 시간: {gen_end_time - gen_start_time:.2f} 초\n")

        print("\n-------------------------------------------------------\n")
