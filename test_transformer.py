from transformers import LlamaTokenizer, LlamaForCausalLM

# 初始化模型和分词器
model_name = "meta-llama/Llama-2-7b-chat-hf"  # 你可以选择其他可用的 LLaMA 模型
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)

# 输入文本
input_text = "Hello, how are you?"
inputs = tokenizer(input_text, return_tensors="pt")

# 生成文本
output = model.generate(**inputs, max_length=50)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("Generated Text:", generated_text)
