import argparse
from nltk.tokenize.treebank import TreebankWordDetokenizer


# Apply the edits of a single annotator to generate the corrected sentences.
def main(args):    
	detokenizer = TreebankWordDetokenizer()
	m2 = [m.strip() for m in open(args.m2_file).read().strip().split("\n\n") if m and not m.isspace()]
	out = open(args.out, "w")
	out_tok = open('%s.tokenized' % args.out, "w")
	out_ori = open(args.out_ori, "w")
	out_ori_tok = open('%s.tokenized' % args.out_ori, "w")
	# Do not apply edits with these error types
	skip = {"noop", "Um"}
	
	for sent in m2:
		sent = sent.split("\n")
		cor_sent = sent[0].split()[1:] # Ignore "S "
		out_ori.write(detokenizer.detokenize(sent[0][2:].split()) + "\n")
		out_ori_tok.write(' '.join(sent[0][2:].split()) + "\n")
		edits = sent[1:]
		offset = 0

		for edit in edits:
			edit = edit.split("|||")			
			if edit and len(edit) > 1 and edit[1] in skip: continue # Ignore certain edits
			coder = int(edit[-1])
			if coder != args.id: continue # Ignore other coders
			span = edit[0].split()[1:] # Ignore "A "
			start = int(span[0])
			end = int(span[1])
			cor = edit[2].split()
			cor_sent[start+offset:end+offset] = cor
			offset = offset-(end-start)+len(cor)
		out.write(detokenizer.detokenize(cor_sent)+"\n")
		out_tok.write(' '.join(cor_sent).strip()+"\n")


if __name__ == "__main__":
	# Define and parse program input
	parser = argparse.ArgumentParser()
	parser.add_argument("m2_file", help="The path to an input m2 file.")
	parser.add_argument("-out", help="A path to where we save the output corrected text file.", required=True)
	parser.add_argument("-out_ori", help="A path to where we save the output orignal text file.", required=True)    
	parser.add_argument("-id", help="The id of the target annotator in the m2 file.", type=int, default=0)	
	args = parser.parse_args()
	main(args)