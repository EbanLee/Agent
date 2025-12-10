import os
import abc
import textwrap
from typing import Optional
from datetime import datetime

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
        return f"""
        You are the Final Answer Generator.

        Your job:
        - Produce a natural-language answer for the user based on the provided information.
        - Do NOT plan actions, call tools, or mention tools, system logic, or internal steps.
        - Never reveal chain-of-thought, JSON, or internal reasoning.
        - Respond in the user's language.
        - If information is incomplete, answer carefully and avoid hallucinating.

        Your only goal is to give the best final answer to the user.
        """

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
        self.system_prompt = self.build_system_prompt()
        
        print(f"Control {model_name=}\n")

    def build_system_prompt(self) -> str:
        tool_str = '\n'.join([f"- {name}: {tool.description}" for name, tool in self.tools.items()])
        prompt =  f"""
        You are the controller (Reasoner) of a tool-using agent.
        You have access to the following tools:

        {tool_str}

        Your role:
        - Look at the user's request (and optional chat history).
        - Decide whether to call a tool to handle the request.
        - Never write the final answer for the user.
        
        Output format:
        - You MUST output exactly ONE valid JSON object and NOTHING else.
        - If you include anything outside the JSON object, the output is invalid.
        - The JSON must always have this structure:

        {{
            "thought": "In 1–3 sentences in the user's language, describe which tool you will call next (or 'finish') and why. Do NOT include the final answer or any detailed factual claims.",
            "action": "one of the tool names above, or 'finish'",
            "action_input": {{ the parameters to pass to the tool (dict) }}
        }}

        Rules:
        1. Use tools only when the user’s request requires information, actions, or operations that cannot be completed through reasoning alone.
          - If a part of the request can be answered or handled directly (including using information from recent conversation), do NOT call a tool for that part.
          - Use tools only for the specific parts of the request that actually require them.
        2. Only one tool call per step.
          - If you need to call multiple tools, call them one by one in separate steps.
          - After each tool call, wait for the OBSERVATION before deciding the next step.
        3. "finish" means:
          - You already have enough information to let the final answer generator respond to the user.
          - No additional tool calls are required.
        4. Task decomposition:
          - If the user's request or instruction contains multiple entities, items, or subtasks, you MUST break the work into smaller steps.
          - Even if the user does NOT explicitly request separation, decompose the task whenever multiple reasoning or tool-usage steps are logically required.
          - When the request asks about multiple entities, call tools separately for each main entity (e.g., handle 'A' first, then 'B', not "A and B" in one query).
          - Follow this pattern: Step 'A' → observe about 'A'. → Step 'B' → observe about 'B' → ... → finish.
        5. When using the web_search tool (or any search-based tool):
          - Keep the query short, keyword-based, and focused.
          - Do not combine multiple main entities in a single query; search for each entity separately (e.g., "A", then "B").
          - If the OBSERVATION does not contain the important information you need, refine the query and search again.
          - For time-sensitive or externally changing information (e.g., real-world facts, current status, events), prefer using a search tool instead of relying on prior knowledge.
        6. Use the same language as the user's request. Use English only when needed (e.g., technical terms, English search keywords).
        7. The "action_input" must ALWAYS be a JSON object (dictionary), even if it is empty.
          Example: {{ "query": "A's current stock price" }} or {{}}.

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
        messages = [{'role': 'system', 'content': self.system_prompt}] + messages[max(0, len(messages)-(self.remember_turn*2)):] + [{'role': "user", 'content': f"- Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n" + user_input}]
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
                temperature=0.2
            )
            generated_output = outputs[0][len(inputs.input_ids[0]):].tolist()
            try:
                index = len(generated_output) - generated_output[::-1].index(self.think_token_id)
            except ValueError:
                index = 0

            output_context = self.tokenizer.decode(generated_output[index:], skip_special_tokens=True)
            print(f"\n-----------------------------------------\n에이전트 판단: \n{output_context}\n-----------------------------------------\n")

            try:
                output_dict = load_json(output_context)
            except json.JSONDecodeError:
                # JSON형태가 아닐 때 다시요청
                observation = None
                messages+=[{'role': 'assistant', 'content': f"{output_context}"}, {'role': 'user', 'content': f"You MUST output exactly ONE valid JSON object and NOTHING else. Please answer again."}]
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
                result.append(f"thought={thought} \naction={action} \nOBSERVATION: \n{observation}\n")
            except TypeError as e:
                observation = f"\naction={action} \naction_input={action_input} \nOBSERVATION: \n[tool call error] {e}\n\n Please answer again."
            except Exception as e:
                observation = f"\naction={action} \naction_input={action_input} \nOBSERVATION: \n[tool call error] {e}\n\n Please answer again."
            
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
        reasoner_result_str = self.reasoner.generate(user_input, self.history)
        print("\n ----------------------------------- reasoner 사용 결과 ----------------------------------- \n\n" ,reasoner_result_str, "\n\n ----------------------------------- reasoner 사용 결과 ----------------------------------- \n")
        final_user_input = (
            f"[USER QUESTION]\n{user_input}\n\n"
            f"[TOOL RESULTS]\n{reasoner_result_str}\n\n"
            f"Please provide the final answer to the user's question based on the information above."
        )

        # print(f"\n최종 입력:\n{final_user_input}\n")

        result_output = self.chat_bot.generate(final_user_input, self.history)
        
        self.history.append({'role':'user', 'content': user_input})
        self.history.append({'role':'assistant', 'content': result_output})
        

        return result_output