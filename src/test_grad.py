from undecorated import undecorated
from types import MethodType
from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoModelForSeq2SeqLM
from transformers import T5Config
class Args:
    def __init__(self):
        self.backbone = 't5-small'  # We'll use the 't5-small' model for quick testing

args = Args()

# Create a configuration object for T5. Using default settings for simplicity
config = T5Config()

model_rec = T5ForConditionalGeneration.from_pretrained(args.backbone, config=config)
# generate with gradient
generate_with_grad = undecorated(model_rec.generate)
model_rec.generate_with_grad = MethodType(generate_with_grad, model_rec)

model_rec.to('cuda')

print()