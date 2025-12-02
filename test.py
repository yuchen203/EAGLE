import torch, json, time
from eagle.model.ea_model import EaModel
from fastchat.model import get_conversation_template 

test_name = "gsm8k"
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
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # generate
    start.record()
    output_ids = model.eagenerate(
        input_ids,
        temperature=0.0,
        max_new_tokens=512
    )
    end.record()
    torch.cuda.synchronize()
    duration = start.elapsed_time(end) / 1000.0
    output = model.tokenizer.decode(output_ids[0])
    return output, duration

output_list = []
duration_list = []
total_examples = 100

with open(f"{test_name}_input.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)
for i in range(total_examples):
    output, duration =run_one_question(dataset[str(i)])
    print("finished example", i)
    output_list.append(output)
    duration_list.append(duration)

result = {}
result["dateset"] = test_name
result["model"] = "qwen3_8b_eagle3"
result["total_examples"] = total_examples
result["generated_texts"] = []
for i in range(total_examples):
    result["generated_texts"].append({
        "example_id": i,
        "input_text": dataset[str(i)],
        "full_text": output_list[i]
    })
result["times"] = duration_list

with open(f"{test_name}_eagle3.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=4)