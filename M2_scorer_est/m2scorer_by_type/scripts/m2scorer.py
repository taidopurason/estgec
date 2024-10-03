#!/usr/bin/python

# This file is part of the NUS M2 scorer.
# The NUS M2 scorer is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# The NUS M2 scorer is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# file: m2scorer.py
# 
# score a system's output against a gold reference 
#
# Usage: m2scorer.py [OPTIONS] proposed_sentences source_gold
# where
#  proposed_sentences   -   system output, sentence per line
#  source_gold          -   source sentences with gold token edits
# OPTIONS
#   -v    --verbose             -  print verbose output
#   --very_verbose              -  print lots of verbose output
#   --max_unchanged_words N     -  Maximum unchanged words when extracting edits. Default 2."
#   --beta B                    -  Beta value for F-measure. Default 0.5."
#   --ignore_whitespace_casing  -  Ignore edits that only affect whitespace and caseing. Default no."
#

import sys
import levenshtein
from getopt import getopt
from util import paragraphs
from util import smart_open



def load_annotation_original(gold_file):
    source_sentences = []
    gold_edits = []
    fgold = smart_open(gold_file, 'r')
    puffer = fgold.read()
    fgold.close()
    #puffer = puffer.decode('utf8') # smart_open opens it in utf-8 now
    for item in paragraphs(puffer.splitlines(True)):
        item = item.splitlines(False)
        sentence = [line[2:].strip() for line in item if line.startswith('S ')]
        assert sentence != []
        annotations = {}
        for line in item[1:]:
            if line.startswith('I ') or line.startswith('S '):
                # S - the original sentence, tokenized (whitespaces)
                # I - ?
                continue
            assert line.startswith('A ')
            # A - annotatod mistake: 
            # A 1 2|||NN|||otter|||REQUIRED|||-NONE-|||1
            # A start_offset end_offset|||error_type|||correction|||-|||-|||annotator no.
            line = line[2:]
            fields = line.split('|||')
            start_offset = int(fields[0].split()[0])
            end_offset = int(fields[0].split()[1])
            etype = fields[1] # error type
            if etype == 'noop':
                # noop => no corrections needed, the original sentence is OK
                start_offset = -1
                end_offset = -1
            corrections =  [c.strip() if c != '-NONE-' else '' for c in fields[2].split('||')]
            # NOTE: start and end are *token* offsets
            original = ' '.join(' '.join(sentence).split()[start_offset:end_offset])
            annotator = int(fields[5])
            if annotator not in annotations.keys():
                annotations[annotator] = []
            annotations[annotator].append((start_offset, end_offset, original, corrections))
            # original - String, original words inside the correction range, whitespace-tokenized
            # corrections - String, proposed correction inside the correction range, whitespace-tokenized
        tok_offset = 0
        for this_sentence in sentence:
            # Combine sentence-edits to be equal-length lists - just because the gold standard may have several sentences per unit.
            tok_offset += len(this_sentence.split())
            source_sentences.append(this_sentence)
            this_edits = {}
            for annotator, annotation in annotations.items():
                this_edits[annotator] = [edit for edit in annotation if edit[0] <= tok_offset and edit[1] <= tok_offset and edit[0] >= 0 and edit[1] >= 0]
            if len(this_edits) == 0:
                this_edits[0] = []
            gold_edits.append(this_edits)
    return (source_sentences, gold_edits)


def load_annotation(gold_file):
    source_sentences = []
    gold_edits = []
    fgold = smart_open(gold_file, 'r')
    puffer = fgold.read()
    fgold.close()
    #puffer = puffer.decode('utf8') # smart_open opens it in utf-8 now
    for item in paragraphs(puffer.splitlines(True)):
        # Split at linebreaks (keep linebreaks)
        # paragraphs -> group by separator lines => each 'item' should be S-A* chunk (separated by empty lines in original document)
        item = item.splitlines(False)
        sentence = [line[2:].strip() for line in item if line.startswith('S ')]
        assert sentence != []
        annotations = {}
        for line in item[1:]:
            if line.startswith('I ') or line.startswith('S '):
                # S - the original sentence, tokenized (whitespaces)
                # I - ?
                continue
            assert line.startswith('A ')
            # A - annotatod mistake: 
            # A 1 2|||NN|||otter|||REQUIRED|||-NONE-|||1
            # A start_offset end_offset|||error_type|||correction|||-|||-|||annotator no.
            line = line[2:]
            fields = line.split('|||')
            start_offset = int(fields[0].split()[0])
            end_offset = int(fields[0].split()[1])
            etype = fields[1] # error type
            if etype == 'noop':
                # noop => no corrections needed, the original sentence is OK
                start_offset = -1
                end_offset = -1
            corrections =  [c.strip() if c != '-NONE-' else '' for c in fields[2].split('||')]
            # NOTE: start and end are *token* offsets
            original = ' '.join(' '.join(sentence).split()[start_offset:end_offset])
            annotator = int(fields[5])
            if annotator not in annotations.keys():
                annotations[annotator] = []
            annotations[annotator].append((start_offset, end_offset, original, corrections, etype)) #Added errortype
            # original - String, original words inside the correction range, whitespace-tokenized
            # corrections - String, proposed correction inside the correction range, whitespace-tokenized
            
        # check if there are "WO" error types in an annotator's edits
        no_annotators = len(annotations.keys())-1 #How many annotators do we already have
        #print(no_annotators)
        pseudo_annotations = {}
        for annotator in annotations:
            wo = [edit for edit in annotations[annotator] if edit[4]=="R:WO"]
            #print("WO",wo)
            if len(wo)>0:
                #print("Has WO!")
                for i, this_wo in enumerate(wo):
                    # check if it overlaps with any other edits
                    wo_start = this_wo[0]
                    wo_end = this_wo[1]
                    # let us assume there are no overlapping WO edits
                    overlapping = [edit for edit in annotations[annotator] if edit[0]>=wo_start and edit[1]<=wo_end and edit[4]!="R:WO"]
                    #print("overlapping:",overlapping)
                    # if so, create pseudo-annotators:
                    # A_1 - WO tag, but no others
                    if len(overlapping)>0:
                        # 1. Keep WO change only - everything must be correct!
                        #                          essentially what you would get with combined error types.
                        no_annotators += 1
                        pseudo_annotator = no_annotators
                        pseudo_annotations[pseudo_annotator] = [edit for edit in annotations[annotator] if edit not in overlapping]
                        # ToDo:
                        # Add pseudoedits here to account for all found errors? - now it only counts as one error found
                        # 2. Keep other edits only
                        no_annotators += 1
                        pseudo_annotator = no_annotators
                        pseudo_annotations[pseudo_annotator] = [edit for edit in annotations[annotator] if edit!=this_wo]
                        # 3. Keep WO change, but also allow for other edits not to have taken place...
                        # ToDo
            
        # Back to 4-element edits
        annotations2 = {}
        # KL: for now, let's keep those error types in edits
        for annotator, annotation in annotations.items():
            #annotations2[annotator] = [(start_offset, end_offset, original, corrections) for (start_offset, end_offset, original, corrections, etype) in annotation]
            annotations2[annotator] = [(start_offset, end_offset, original, corrections, etype) for (start_offset, end_offset, original, corrections, etype) in annotation]
        for annotator, annotation in pseudo_annotations.items():
            #annotations2[annotator] = [(start_offset, end_offset, original, corrections) for (start_offset, end_offset, original, corrections, etype) in annotation]
            annotations2[annotator] = [(start_offset, end_offset, original, corrections, etype) for (start_offset, end_offset, original, corrections, etype) in annotation]
        annotations = annotations2
            
        tok_offset = 0
        for this_sentence in sentence:
            # Combine sentence-edits to be equal-length lists - just because the gold standard may have several sentences per 'item' (if you missed an empty line).
            tok_offset += len(this_sentence.split())
            source_sentences.append(this_sentence)
            this_edits = {}
            for annotator, annotation in annotations.items():
                this_edits[annotator] = [edit for edit in annotation if edit[0] <= tok_offset and edit[1] <= tok_offset and edit[0] >= 0 and edit[1] >= 0]
            if len(this_edits) == 0:
                this_edits[0] = []
            gold_edits.append(this_edits)
    return (source_sentences, gold_edits)



def print_usage():
    #Old: print >> sys.stderr, "fatal error"
    #New: print("fatal error", file=sys.stderr)
    print("Usage: m2scorer.py [OPTIONS] proposed_sentences gold_source", file=sys.stderr)
    print("where", file=sys.stderr)
    print("  proposed_sentences   -   system output, sentence per line", file=sys.stderr)
    print("  source_gold          -   source sentences with gold token edits", file=sys.stderr)
    print("OPTIONS", file=sys.stderr)
    print("  -v    --verbose                   -  print verbose output", file=sys.stderr)
    print("        --very_verbose              -  print lots of verbose output", file=sys.stderr)
    print("        --max_unchanged_words N     -  Maximum unchanged words when extraction edit. Default 2.", file=sys.stderr)
    print("        --beta B                    -  Beta value for F-measure. Default 0.5.", file=sys.stderr)
    print("        --ignore_whitespace_casing  -  Ignore edits that only affect whitespace and caseing. Default no.", file=sys.stderr)



#max_unchanged_words=2
max_unchanged_words=15 # Elagu eesti keele sõnaliikumised
beta = 0.5
ignore_whitespace_casing= False
verbose = False
very_verbose = False
opts, args = getopt(sys.argv[1:], "v", ["max_unchanged_words=", "beta=", "verbose", "ignore_whitespace_casing", "very_verbose"])
for o, v in opts:
    if o in ('-v', '--verbose'):
        verbose = True
    elif o == '--very_verbose':
        very_verbose = True
    elif o == '--max_unchanged_words':
        max_unchanged_words = int(v)
    elif o == '--beta':
        beta = float(v)
    elif o == '--ignore_whitespace_casing':
        ignore_whitespace_casing = True
    else:
        print("Unknown option :"+ str(o), file=sys.stderr)
        print_usage()
        sys.exit(-1)

# starting point
if len(args) != 2:
    print_usage()
    sys.exit(-1)

system_file = args[0]
gold_file = args[1]

# load source sentences and gold edits
source_sentences, gold_edits = load_annotation(gold_file)
#source_sentences, gold_edits = load_annotation_original(gold_file)

# load system hypotheses
fin = smart_open(system_file, 'r') # smart_open opens it in utf-8 now
#system_sentences = [line.decode("utf8").strip() for line in fin.readlines()]
system_sentences = [line.strip() for line in fin.readlines()]
fin.close()

#print("proposed:",len(system_sentences))
#print("source:",len(source_sentences))
#print("edits:",len(gold_edits))
#print(gold_edits[-1])
#print()

p, r, f1, stderrs = levenshtein.batch_multi_pre_rec_f1(system_sentences, source_sentences, gold_edits, max_unchanged_words, beta, ignore_whitespace_casing, verbose, very_verbose, bootstrap_n=10000, seed=42)
if stderrs is not None:
    z = 1.959963984540054
    print("Precision_ci   : %.8f" % (stderrs["p"] * z))
    print("Recall_ci      : %.8f" % (stderrs["r"] * z))
    print("F_%.1f_ci       : %.8f" % (beta, stderrs["f1"] * z))
    print("Precision_stderr   : %.8f" % stderrs["p"])
    print("Recall_stderr      : %.8f" % stderrs["r"])
    print("F_%.1f_stderr       : %.8f" % (beta, stderrs["f1"]))

print("Precision   : %.4f" % p)
print("Recall      : %.4f" % r)
print("F_%.1f       : %.4f" % (beta, f1))

