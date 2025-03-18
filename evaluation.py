import torch
import evaluate
from transformer import Transformer
from config import TokenizerConfig, EvaluationConfig, MyDatasets
from utils import prepare_tokenizer, prepare_translation


test_data = MyDatasets.load_test()
bleu = evaluate.load('bleu')
sacrebleu = evaluate.load("sacrebleu")

checkpoint = torch.load(EvaluationConfig.checkpoint_file)
model_config = checkpoint['model_config']
model = Transformer(**model_config)
model.load_state_dict(checkpoint['model_state'])
model.to(EvaluationConfig.device)

tokenizer = TokenizerConfig.load_tokenizer()
tokenizer = prepare_tokenizer(tokenizer)

predictions = []
references = []


for batch in test_data.iter(EvaluationConfig.batch_size):
    batch = prepare_translation(batch)
    source = batch[MyDatasets.source_language]
    target = batch[MyDatasets.target_language]

    pred = model.translate(
        source,
        tokenizer,
        TokenizerConfig.start_token,
        TokenizerConfig.end_token,
        EvaluationConfig.max_tokens,
    )

    predictions += pred
    references += [[sentence] for sentence in target]

bleu_score = bleu.compute(predictions=predictions, references=references)
sacrebleu_score = sacrebleu.compute(predictions=predictions, references=references)
results = {'bleu': bleu_score['bleu'], 'sacrebleu': sacrebleu_score['score']}

