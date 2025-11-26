import torch, json, time
from datasets import load_dataset
from eagle.model.ea_model import EaModel
from fastchat.model import get_conversation_template 

dataset = load_dataset("mbpp", "full")["test"]

model = EaModel.from_pretrained(
    base_model_path="../Qwen3-8B",
    ea_model_path="../qwen3_8b_eagle3",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
    total_token=-1
)

def run_one_question(question):
    # tokenizer encode to ids
    input_ids = model.tokenizer([question]).input_ids
    input_ids = torch.as_tensor(input_ids).cuda()

    # generate
    start = time.perf_counter()
    output_ids = model.eagenerate(
        input_ids,
        temperature=0.0,
        max_new_tokens=512
    )
    end = time.perf_counter()
    duration = end - start
    output = model.tokenizer.decode(output_ids[0])
    return output, duration

output_list = []
duration_list = []

for i  in range(11, 111):
    for item in dataset:
        if item['task_id'] == i:
            output, duration =run_one_question(item["text"])
            print("finished example", item['task_id'])
            output_list.append(output)
            duration_list.append(duration)

result = {}
result["dateset"] = "mbpp"
result["model"] = "qwen3_8b_eagle3"
result["total_examples"] = 100
result["generated_texts"] = []
for i in range(100):
    result["generated_texts"].append({
        "example_id": i + 11,
        "full_text": output_list[i]
    })
result["times"] = duration_list

with open("mbpp_eagle3.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)