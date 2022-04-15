import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from time import time
from sacremoses import MosesTokenizer, MosesDetokenizer
import spacy
doc = spacy.load('en_core_web_sm')
import jieba.posseg as pseg
import jieba

langs = {'de_DE': 'de', 'en_XX': 'en', 'cs_CZ': 'ck', 'ru_RU': 'ru', 'ar_AR': 'ar', 'ro_RO': 'ro', 'zh_CN': 'ch'}
        
def evaluate_langs_and_save(model_name='mbart', checkpoint_idx=None, lang_codes='de_DE,en_XX,cs_CZ,ru_RU,ar_AR', batch_size=20, data_set_type='dev'):  
    model = AutoModelForSeq2SeqLM.from_pretrained("%s/best_tfmr" % model_name).cuda().eval()
    if model.config.model_type == 'mbart':
        tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25", use_fast=False) 
    else:
        tokenizer = AutoTokenizer.from_pretrained("%s/best_tfmr" % model_name, use_fast=False) 

	
    for lang_code in lang_codes.split(','):
        if model.config.model_type == 'mbart':
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids([lang_code])[0]
        lang_id = langs[lang_code]
        lang_tokenizer = MosesTokenizer(lang_id)                    
        with open('%s/%s-%s.src' % (lang_id, lang_id, data_set_type), encoding='utf-8') as dev_data:
            original_sentences = [l for l in dev_data.readlines()]                        
        corrected_input = []        
        for idx in range(0, len(original_sentences), batch_size):
            if model.config.model_type == 'mbart':
                encoded_input = tokenizer.prepare_seq2seq_batch(original_sentences[idx:idx+batch_size], padding=True, return_tensors='pt', src_lang=lang_code, tgt_lang=lang_code)['input_ids']
            else:
                encoded_input = tokenizer.prepare_seq2seq_batch(original_sentences[idx:idx+batch_size], padding=True, return_tensors='pt')['input_ids']

            generated_ids = model.generate(encoded_input.cuda(), max_length=256)
            [corrected_input.append(l) for l in tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)]

        target_file_name = '%s/%s-%s.cor.%s' % (lang_id, lang_id, data_set_type, checkpoint_idx) if checkpoint_idx else '%s/%s-%s.cor' % (lang_id, lang_id, data_set_type)
        if lang_id == 'en':            
            with open(target_file_name, mode='w+', encoding='utf-8') as input_file:
                [input_file.write('%s\n' % ' '.join([t.text for t in doc('%s\n' % l)][:-1])) for l in corrected_input]
        else:
            if lang_id == 'ch':
                with open(target_file_name, mode='w+', encoding='utf-8') as input_file:
                    [input_file.write('%s\n' % ' '.join([t[0] for t in jieba.tokenize(l)]).replace(',', '\uff0c')) for l in corrected_input]
            else:		
                with open(target_file_name, mode='w+', encoding='utf-8') as input_file:
                    [input_file.write('%s\n' % ' '.join(lang_tokenizer.tokenize(l))) for l in corrected_input]
            
        

if __name__ == "__main__":
    t0 = time()
    if len(sys.argv) > 1:
        evaluate_langs_and_save(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5])
    else:
        evaluate_langs_and_save(sys.argv[1])
    print(time() - t0)