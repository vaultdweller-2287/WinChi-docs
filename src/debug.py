from ctransformers import AutoModelForCausalLM

model_path = r"C:\Users\jaibj\Downloads\phi-3-mini-4k-instruct.gguf"

print("Starting script...")

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    model_type="llama",  # llama/llama2 etc
    gpu_layers=0         # set >0 if you have GPU support
)
print("Model loaded.")

print("Generating...")
# simple blocking generation
response = model("Explain gravity simply.", max_new_tokens=50)

print("Response:")
print(response)

print("Done.")
