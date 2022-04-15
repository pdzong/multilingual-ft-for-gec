import sys

def filter_empty_sentences(lang_id):
    with open('%s/train.source' % lang_id) as data:
        src_lines = list(data.readlines())
    with open('%s/train.target' % lang_id) as data:
        trg_lines = list(data.readlines())
    with open('%s/train.source' % (lang_id), mode='w+') as src_input_data:
        with open('%s/train.target' % (lang_id), mode='w+') as trg_input_data:
            for s, t in zip(src_lines, trg_lines):
                if s and t and not s.isspace() and not t.isspace():
                    src_input_data.write(s)
                    trg_input_data.write(t)

if __name__ == "__main__":
    filter_empty_sentences(sys.argv[1])