import time

import torch
from torch.utils.data import Dataset, DataLoader

from model import model_class
import tools
from tools import search_tool
from utils import file_utils

test_dataset_path = "./dataset/search_dataset.json"
test_dataset = file_utils.read_json(test_dataset_path)
print(test_dataset[0])
print(type(test_dataset[0]))


total_tools: dict[str, tools.Tool] = {
    search_tool.WebSearchTool.name: search_tool.WebSearchTool()
}
# agent = model_class.Agent(total_tools=total_tools)
reasoner = model_class.Reasoner(total_tools=total_tools, remember_turn=0)
reasoner.model.eval()

torch.cuda.empty_cache()
with torch.no_grad():
    for i, data in enumerate(test_dataset):
        id = data["id"]
        question = data["question"]  # user input
        history = data["history"]
        search_required = data["search_required"]  # label
        search_type = data["search_type"]

        reasoner_out = reasoner.generate(
            data["question"],
        )
