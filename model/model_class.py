import os
import abc
import textwrap
from typing import Optional
from datetime import datetime
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from utils import file_utils
from utils.functions import *
import tools

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{DEVICE=}", "\n")

curr_path = os.path.dirname(__file__)
model_config = file_utils.read_yaml(os.path.join(curr_path, "model_config.yaml"))
# print(model_config)
quant_config = BitsAndBytesConfig(load_in_8bit=True)

class LLM(abc.ABC):
    """
    LLM 부모 클래스
    """
    def __init__(self, model_name: str, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map=DEVICE,
            # trust_remote_code=True,
        )

    @abc.abstractmethod
    def generate(self, **kwargs):
        pass

    # def input_truncate(self, ):

class ChatBot(LLM):
    """
    대화 챗봇 Class
    """

    def __init__(self, model_name = model_config["chat_model_name"], remember_turn: int=2):
        """
        챗봇 모델과 토크나이저 설정.
        <think> 토큰 있으면 이후 답변을 위해 
        """
        super().__init__(model_name)
        # self.history = []
        self.think_token_id = self.tokenizer("</think>").input_ids
        self.think_token_id = self.think_token_id[0] if len(self.think_token_id)==1 else -1
        self.remember_turn = remember_turn

        print(f"Chat {model_name=}\n")
        
        # # ---------------------------------------------------
        # print("cuda available:", torch.cuda.is_available())
        # print("model main device:", next(self.model.parameters()).device)

        # try:
        #     print("hf_device_map:", self.model.hf_device_map)
        # except AttributeError:
        #     print("no hf_device_map attr")
        # # ---------------------------------------------------

    def build_system_prompt(self):
        prompt = f"""
You are the Final Answer Generator.

Your job:
- Provide a natural-language answer using ONLY the given information.
- Do NOT call tools, plan actions, or mention internal reasoning.
- If information is insufficient, state that clearly rather than guessing.

Language:
- You must respond in the same language as "[USER REQUEST]".
"""
        return textwrap.dedent(prompt).strip()

    def generate(self, user_input: str, history: list) -> str:
        messages = history[1:] if history and history[0]['role']=='system' else history[:]
        messages = [{'role': 'system', 'content': self.build_system_prompt()}] + messages[max(0, len(messages)-(self.remember_turn*2)):] + [{'role': "user", 'content': user_input}]
        
        # system 있으면 넣어주기
        if history and history[0]['role']=='system' and messages[0]['role']!='system':
            messages = [history[0]] + messages

        # 자동 템플릿 생성(sos, eos 등)
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        print(f"\n------------------------ 최종 입력 ------------------------\n\n{text}\n\n--------------------------------------------------------------\n")

        inputs = self.tokenizer(text, return_tensors='pt')
        # print(inputs)
        outputs = self.model.generate(
            **inputs.to(DEVICE),
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=512,    # 32768
            )
        generated_output = outputs[0][len(inputs.input_ids[0]):].tolist()
        print(f"{len(generated_output)=}\n")
        
        # decoder는 질문과 답변 모두 뱉으니까 답변에 대한 index 계산
        try:
            index = len(generated_output) - generated_output[::-1].index(self.think_token_id)
        except ValueError:
            index = 0

        output_context = self.tokenizer.decode(generated_output[index:], skip_special_tokens=True)

        # self.history.append({'role': "user", "content": user_input})
        # self.history.append({'role': "assistant", "content": output_context})

        return output_context

class Reasoner(LLM):
    """
    답변할 수 있는 상태로 만들어주는 컨트롤러
    """
    def __init__(self, model_name = model_config["reasoning_model_name"], total_tools: dict[str, tools.Tool] = {}, remember_turn: int=3):
        super().__init__(model_name)
        self.think_token_id = self.tokenizer("</think>").input_ids
        self.think_token_id = self.think_token_id[0] if len(self.think_token_id)==1 else -1
        self.remember_turn = remember_turn
        self.tools = total_tools
        
        print(f"Control {model_name=}\n")

    def build_system_prompt(self, user_input) -> str:
        tool_str = '\n'.join([f"- {name}: {tool.description}" for name, tool in self.tools.items()])
        prompt =  f"""
You are the controller (Reasoner) of a tool-using agent.
You NEVER answer the user's question directly. You ONLY decide whether to call a tool or finish.

You have access to the following tools:
{tool_str}

Output format (VERY IMPORTANT):
- You MUST respond with exactly ONE JSON object and NOTHING else.
- The JSON MUST start with '{{' and end with '}}'.
- Valid schema:
{{
"thought": "In 1–3 sentences, explain whether you need to call a tool or can set 'action' to 'finish', and why.",
"action": "<tool name or 'finish'>",
"action_input": {{ parameters. (Use {{}} if there are no arguments.) }}
}}

Rules:

1. Task decomposition:
- Break the user request into each mentioned target (e.g., each person, object, or information item).
- For each target, independently check whether the required information already exists in the conversation history.
- Do NOT infer or expand subtasks. Only handle exactly what the user requested.
- Do NOT merge multiple requested entities into a single search query.

2. Call tools only when needed.
- If part of your request requires a tool, call the tool ONLY for that part.
- If the request can be answered directly from your own general knowledge without any external information, do NOT call a tool and set "action": "finish".
- For information tools (e.g., web search): use ONLY if the needed information is NOT already in recent conversation. Do NOT re-search info already known.
- For action tools (e.g., git, file operations): ALWAYS perform the action when the user requests it, even if similar actions were done before.

3. "finish" means the final answer generator already has enough information.

4. Web search rules:
- Queries must be short, keyword-based, and focused.
- Refine and retry only if OBSERVATION lacks required information.

5. Language:
- You must respond in the same language as "{user_input}".
- You may use English only for short technical terms, tool names, or search keywords when necessary.
- Never use any other language.

6. One tool call per step.
- After a tool call, wait for OBSERVATION and decide the next step.

7. "action_input" must ALWAYS be a JSON object (even if empty).

Follow these rules strictly.
"""

        return textwrap.dedent(prompt).strip()


    def generate(self, user_input: str, history: list, max_repeat_num: int=5) -> str:  
        """
        답변할 수 있는 상태 or 최대 반복 횟수까지 tool 사용.

        args:
            user_input: 유저가 입력한 질문(명령)
            history: 이때까지 진행된 대화 기록
            max_repeat_num: tool을 최대 몇번까지 사용할지

        returns:
            str: tool사용해서 나온 결과
        """
        result = []
        messages = history[1:] if history and history[0]['role']=='system' else history[:]
        messages = [{'role': 'system', 'content': self.build_system_prompt(user_input)}] + messages[max(0, len(messages)-(self.remember_turn*2)):] + [{'role': "user", 'content': f"[Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n\n" + user_input}]
        observation: Optional[str] = None

        for _ in range(max_repeat_num):
            if observation is not None:
                messages.append({'role': 'user', 'content': f"OBSERVATION: \n{observation}"})

            # 자동 템플릿 생성(sos, eos 등)
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = self.tokenizer(text, return_tensors='pt')
            outputs = self.model.generate(
                **inputs.to(DEVICE),
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=256,
                temperature=0.1
            )
            generated_output = outputs[0][len(inputs.input_ids[0]):].tolist()
            try:
                index = len(generated_output) - generated_output[::-1].index(self.think_token_id)
            except ValueError:
                index = 0

            output_context = self.tokenizer.decode(generated_output[index:], skip_special_tokens=True)
            print(f"\n-----------------------------------------\nReasoner 판단: \n{output_context}\n-----------------------------------------\n")

            try:
                output_dict = load_json(output_context)
            except json.JSONDecodeError:
                # JSON형태가 아닐 때 다시요청
                observation = None
                messages+=[{'role': 'assistant', 'content': f"{output_context}"}, {'role': 'user', 'content': f"Invalid format. Respond again with ONLY one JSON object, nothing else."}]
                continue

            messages+=[{'role': "assistant", "content": output_context}]
            
            action = output_dict.get("action")
            if action.strip().lower()=="finish":
                break

            action_input: dict[str, str] = output_dict.get("action_input")
            thought = output_dict.get("thought")

            # 도구 사용 성공했을 때 만 result에 저장.
            try:
                tool: tools.Tool = self.tools[action]
                observation = tool(**action_input)
                result.append(f"thought={thought} \naction={action} \nOBSERVATION= \n{observation}\n")
            except TypeError as e:
                observation = f"\naction={action} \naction_input={action_input} \nOBSERVATION= \n[tool call error] {e}\n\n Please answer again."
            except Exception as e:
                observation = f"\naction={action} \naction_input={action_input} \nOBSERVATION= \n[tool call error] {e}\n\n Please answer again."
            
            print("Reasoner OBSERVATION: \n", observation, "\n")
        print()
        
        if not result:
            return output_dict["thought"]

        # result = '\n'.join(result)

        return result[-1]

class Agent:
    def __init__(self, total_tools:dict={}, remember_turn:int=3):
        self.tools = total_tools
        self.chat_bot = ChatBot(remember_turn=remember_turn)
        self.chat_bot.model.eval()        
        self.reasoner = Reasoner(total_tools=self.tools, remember_turn=remember_turn)
        self.reasoner.model.eval()

        self.history = []

    def run(self, user_input: str):
        Reasoner_start_time = time.time()
        reasoner_result_str = self.reasoner.generate(user_input, self.history)
        Reasoner_end_time = time.time()

        print("\n ----------------------------------- reasoner 사용 결과 ----------------------------------- \n\n" ,reasoner_result_str, "\n\n ----------------------------------- reasoner 사용 결과 ----------------------------------- \n")
        final_user_input = (
            f"[USER REQUEST]: \n{user_input}\n\n"
            f"[TOOL RESULTS]: \n{reasoner_result_str}\n\n"
            f"Please provide the final answer to the user's question based on the information above."
        )
        # print(f"\n최종 입력:\n{final_user_input}\n")

        final_model_start_time = time.time()
        result_output = self.chat_bot.generate(final_user_input, self.history)
        final_model_end_time = time.time()
        print(f"\nReasoner 생성 시간: {Reasoner_end_time - Reasoner_start_time:.2f} 초\n")
        print(f"\nFinal Model 생성 시간: {final_model_end_time - final_model_start_time:.2f} 초\n")

        self.history.append({'role':'user', 'content': user_input})
        self.history.append({'role':'assistant', 'content': result_output})
        

        return result_output