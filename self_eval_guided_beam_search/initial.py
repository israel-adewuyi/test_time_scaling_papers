import sglang as sgl

# Define a generation function using SGLang primitives
@sgl.function
def generate_story():
    # Start generation with a prompt
    prompt = "Once upon a time,"
    # Use `gen` to generate up to 50 new tokens with temperature 0.7
    story = sgl.gen(prompt, max_new_tokens=50, temperature=0.7)
    return story

def main():
    # Run the generation function
    output = sgl.run(generate_story)
    print("Generated output:")
    print(output)

if __name__ == "__main__":
    main()

# import torch
# print(torch.version.cuda)
