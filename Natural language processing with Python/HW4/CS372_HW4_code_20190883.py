
import nltk
from nltk.corpus import conll2000
from nltk import word_tokenize
import csv
import pickle
"""
                The whole idea for implementation: 
The main issue of this homework is the step 3, that after choosing 80 sentences 
with annotated triples, we have to develop a relation extraction model based on 
these sentences. My solution for this issue is: Firstly, I find way for chunking 
each sentence into a tree of chunks. To do this, I create a classifier-based chunker
and train it by the corpus conll2000 in nltk. In fact, this classifier-based chunker 
is introduced in chapter 7 of the textbook, and it is based just on nltk, therefore, 
this chunker is not an external model, however, I use it because it seems to work quite well. 
After training, now I have a chunker to use for next steps. In this homework, I 
choose 5 verbs and form a list:  V= [ activate, inhibit, bind, prevent, accelerate ]. 
Secondly, we will create dataset for training by using 80 sentences that we have chosen. 
The first step is to create an empty list named train_data. The idea is that, from 
a sentence, we will collect all possible triples, and then with each triple M, we 
will extract some of its features to create a dictionary named dict_fea, and finally, 
by looking at the list of correct triples that we have manually annotated for 
that sentence, we will know whether triple M is a correct triple of the form <X,ACTION,Y> 
or not. If triple M is a correct triple, then we append the couple (dict_fea, 1) to 
the list train_data, else we append the couple (dict_fea, 0) to the list_train_data 
( 1 is the label for correct triple and 0 is the label for incorrect triple). 
Thus, we have 2 main problems: firstly, from a sentence, we have to find all possible 
triples from it, and secondly, with each triple, we have to extract good features of 
that triple for training. To solve the first problem, with each sentence S among 
these 80 sentences, we will use the chunker to create a tree of chunks for that sentence. 
Next, we determine all occurrence of each of 5 verbs in list V in the sentence S. 
Assume that we have V1,V2,… Vn appearing in S that each V1 , V2 , … Vn is a verb in 
the list V=[ activate, inhibit, bind, prevent, accelerate ]  or reflected form of 
that verb. We then create an empty list named list_triples to collect all possible 
triples. In previous step, we have used the chunker to create a tree of chunks for 
sentence S, assume the name of this tree is T, then with each verb Vi defined above, 
we use the tree T to determine noun phrases surrounding Vi in sentence S. 
Assume noun_phrase1, noun_phrase2 are 2 noun phrases that appear before Vi in sentence 
S and they are closest to Vi. Similarly, assuming noun_phrase3, noun_phrase4 are 2 noun 
phrases that appear after Vi in sentence S and they are closet to V¬i . Then we have 4 
possible triples (noun_phrase1, Vi , noun_phrase3), (noun_phrase1, Vi, noun_phrase4), 
(noun_phrase2, Vi, noun_phrase3), (noun_phrase2, Vi , noun_phrase4). By doing this for 
all verb V1,V2, … Vn , we can create a list of possible triples for sentence S, and we 
append all of these triples to list_triples. Next, from each triple (A,B,C) from list_triples, 
we have to extract its feature. In the code implementation, I extract some of features 
of (A,B,C) that I think these features are reasonable. For example, we know that A and C 
are noun phrases, and I choose the first feature to be the distance between A and C in the 
tree of chunks of S. I choose the second feature to be a true/false variable, it would be 
True if there is no noun phrase between A and C in the tree of chunks and it would be False 
if there is at least one. There are some other features. From now on, we have collected 
all possible triples and create a dictionary of features for each triple. With each triple, 
we append the couple (dictionary of features of triple, label of triple) to the list 
train_set and now, we have the train dataset. I create “classifier = nltk.MaxentClassifier” to 
use for training. After this training step, with a new sentence K, I would give all 
predicted triples <X,ACTION,Y> annotations to it as following: firstly, creating an 
empty list name predicted_triples. Next, collecting all possible triples from K and create 
a dictionary of features for each triple by doing in the same way as the above part. 
Then, with each triple, we use its dictionary of features as the input for the classifier 
that we have trained, and this classifier would output the label 0 or 1. If the output label 
is 1, then we would append this triple to the list predicted_triples. By doing this, we 
collect all triples predicted by our module for a sentence. To access the performance of 
80 train sentences and 20 test sentences, we use the triples annotations that we have manually 
annotated for each sentence and use the formula: precision = (number of true prediction)/(number of predicted triples), 
recall = (number of true predictions)/( number of true relation) and F1score is the harmonic mean of precision and recall.

"""


"""
firstly, we need to create a classifier-based chunker and train it using the corpus conll2000
function npchunk_features, class ConsecutiveNPChunkTagger, class ConsecutiveNPChunker are functions
and classes that would be use to create the chunker
"""

def npchunk_features(sentence, i, history):
        word, pos = sentence[i]
        return {"pos": pos}
class ConsecutiveNPChunkTagger(nltk.TaggerI):
    def __init__(self, train_sents):
        train_set = []
        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = npchunk_features(untagged_sent, i, history)
                train_set.append( (featureset, tag) )
                history.append(tag)
        self.classifier = nltk.MaxentClassifier.train(
            train_set, algorithm='gis', trace=0)

    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featureset = npchunk_features(sentence, i, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)

class ConsecutiveNPChunker(nltk.ChunkParserI):
    def __init__(self, train_sents):
        tagged_sents = [[((w,t),c) for (w,t,c) in
                         nltk.chunk.tree2conlltags(sent)]
                        for sent in train_sents]
        self.tagger = ConsecutiveNPChunkTagger(tagged_sents)

    def parse(self, sentence):
        tagged_sents = self.tagger.tag(sentence)
        conlltags = [(w,t,c) for ((w,t),c) in tagged_sents]
        return nltk.chunk.conlltags2tree(conlltags)

"""
now, we would train the chunker by the corpus conll2000
"""
train_sents = conll2000.chunked_sents('train.txt')
chunker = ConsecutiveNPChunker(train_sents)

# string_to_list_triples has the input str that str is a string
# this function is to convert a string of the form '<A1,B1,C1> <A2,B2,C2> ...' into
# a list of triple : [('A1','B1','C1'),('A2','B2','C2')...]
def string_to_list_triples(str):
    triples_of_str=[]
    start_=str.find("<")
    while start_!=-1:
        end_= str.find(">",start_)
        triple_str= str[start_+1 : end_]
        tam= triple_str.split(",")  # it is a list
        # str3.lstrip(' ')
        ele_a= tam[0].lstrip(' ')
        ele_a1= ele_a.rstrip(' ')
        ele_b = tam[1].lstrip(' ')
        ele_b1 = ele_b.rstrip(' ')
        ele_c = tam[2].lstrip(' ')
        ele_c1 = ele_c.rstrip(' ')
        triples_of_str.append((ele_a1,ele_b1,ele_c1))
        start_= str.find("<",end_+1)
    return triples_of_str

# collect_csv_content would read data from a csv file
def collect_csv_content(path):
    collect_content_list=[]
    with open (path) as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            collect_content_list.append(list(row))
    return collect_content_list

"""
now, we read data from a csv file into a list 
"""
tam_contents = collect_csv_content("CS372_HW4_output_20190883.csv")[1:]  # [[sentence,[<X,ACTION,Y>,<X,ACTION,Y>...], x2,x3,x4,x5],...]
contents= [[w[0],string_to_list_triples(w[1]),w[3],w[4],w[5],w[6]] for w in tam_contents]
# contents [[sentence1,[(X,ACTION,Y),(X,ACTION,Y)...],x2,x3,x4,x5], [sentence2,[(X,ACTION,Y),(X,ACTION,Y)...],x2,x3,x4,x5], ... ]
csv_sentences = [w[0] for w in contents]  # a list of string

# this function modifies some part-of-speech postag that nltk.pos_tag give wrong tags
def modify_tag(pair):
    if pair[0]=="ligands":
        return (pair[0],"NNS")
    if pair[0]=="exerts":
        return (pair[0],"VBZ")
    if pair[0]=="pathways":
        return (pair[0],"NNS")
    if pair[0]=="chiefly":
        return (pair[0],"RB")
    if pair[0]=="lipoproteins":
        return (pair[0],"NNS")
    if pair[0]=="that":
        return (pair[0],"IN")
    if pair[0]== "amphetamine":
        return (pair[0],"NN")
    if pair[0]== "ligases":
        return (pair[0],"NNS")
    if pair[0]=="sites":
        return (pair[0],"NNS")
    if pair[0]=="ssDNA":
        return (pair[0],"NN")
    if pair[0]=="amnesia":
        return (pair[0],"NN")
    if pair[0]=="insect":
        return (pair[0],"NN")
    if pair[0]=="motilin":
        return (pair[0],"NN")
    if pair[0]=="such":
        return (pair[0],"IN")
    return pair

# filter_tokens function remove '(' and ')' from a tokens
def filter_tokens(tokens):
    exclude_= ["(",")"]
    new_tokens=[w for w in tokens if w not in exclude_]
    return new_tokens

# chunk_one_sentence would sent as input where sent is a string
# it would return a tree of chunks for that sentence
def chunk_one_sentence(sent):  # it is a string
    grammar = r"""
                  NP: {<NP><CC*><NP*>}          # Chunk sequences of DT, JJ, NN
                  """
    cp = nltk.RegexpParser(grammar, loop=5)
    tokens = word_tokenize(sent)

    new_tokens = filter_tokens(tokens)
    sent_pos = nltk.pos_tag(new_tokens)
    sent_pos2 = []  # a list of list
    for pair in sent_pos:
        if pair[0] not in ACTION_reflects:
            sent_pos2.append(modify_tag(pair))
        else:
            if pair[1].startswith("V"):
                sent_pos2.append(pair)
            else:
                if pair[0].endswith("ed") or pair[0] == "bound":
                    sent_pos2.append((pair[0], "VBD"))
                elif pair[0].endswith("s"):
                    sent_pos2.append((pair[0], "VBZ"))
                else:
                    sent_pos2.append((pair[0], "VBP"))
    return cp.parse(chunker.parse(sent_pos2))


ACTION_reflects= ["activate", "activated", "activates", "inhibit", "inhibited", "inhibits","bind", "binds", "bound",  "prevent", "prevented", "prevents", "accelerate", "accelerates","accelerated" ]
ACTION_dict ={"activate": ["activate", "activated", "activates"], "inhibit": ["inhibit", "inhibited", "inhibits"], "bind": ["bind", "binds", "bound"], "stimulate": ["stimulate" , "stimulated", "stimulates"], "prevent": ["prevent", "prevented", "prevents"]}

# count_occurence_in_tokens function would take as input a string word
# and a list of string named tokens, then it counts the number of occurrence
# of word inside tokens
def count_occurence_in_tokens(word,tokens):
    count=0
    for w in tokens:
        if w==word:
            count+=1
    return count

# get_noun_chunk function take sent as input where sent is a string of sentence
# it would return all possibile triples of the form <X,ACTION,Y> from sent as well
# as some useful information according to each triple
def get_noun_chunk(sent): # intput is a string, as a sentence
    noun_chunks_before_action_indexs=[]
    noun_chunks_after_action_indexs=[]
    noun_chunks_before_action_strs = []
    noun_chunks_after_action_strs = []
    chunk_tree= chunk_one_sentence(sent)
    action_chunk_index_list=[]
    ACTION=[]
    for i in range(len(chunk_tree)):
        try:
            leaves_tree = chunk_tree[i].leaves()
            leaves_str = [w[0] for w in leaves_tree]
            for w in leaves_str:
                if w in ACTION_reflects:
                    ACTION.append(w)
                    action_chunk_index_list.append(i)
        except:
            pass
    list_all_verb_and_its_noun_chunks=[]
    for m in range(len(ACTION)):
        action_chunk_index = action_chunk_index_list[m]  # get the first ACTION verb in the sentence
        action_chunk_verb = ACTION[m]
        for i in range(action_chunk_index):
            try:
                if chunk_tree[i].label() == "NP":
                    noun_chunks_before_action_indexs.append(i)
            except:
                pass
        for i in range(action_chunk_index + 1, len(chunk_tree)):
            try:
                if chunk_tree[i].label() == "NP":
                    noun_chunks_after_action_indexs.append(i)
            except:
                pass
        action_position_in_sentence = sent.find(action_chunk_verb)
        for index in noun_chunks_before_action_indexs:
            t = chunk_tree[
                index]  # this is a tree, like this : (NP (NP various/JJ particles/NNS) and/CC (NP nanomedicines/NNS))
            tokens_tuple_t = t.leaves()  # [('various', 'JJ'), ('particles', 'NNS'), ('and', 'CC'), ('nanomedicines', 'NNS')]

            tokens_t = [pair[0] for pair in tokens_tuple_t]
            start_pos = sent.rfind(tokens_t[0], 0, action_position_in_sentence)
            end_pos = sent.rfind(tokens_t[-1], start_pos, action_position_in_sentence)
            str1 = sent[max(start_pos - 1, 0): end_pos]
            str2 = sent[end_pos: min(action_position_in_sentence, end_pos + len(tokens_t[-1]) + 1)]
            str3 = str1 + str2
            str4 = str3.lstrip(' ')
            str5 = str4.rstrip(' ')
            str6 = str5 if not (str5.endswith(",") or str5.endswith(".") or str5.endswith(";")) else str5[:-1]
            noun_chunks_before_action_strs.append(str6)
        for index in noun_chunks_after_action_indexs:
            t = chunk_tree[index]
            tokens_tuple_t = t.leaves()
            tokens_t = [pair[0] for pair in tokens_tuple_t]
            start_pos = sent.find(tokens_t[0], action_position_in_sentence + 1, len(sent) - 1)
            end_pos = sent.find(tokens_t[-1], start_pos, len(sent) - 1)
            str1 = sent[max(start_pos - 1, action_position_in_sentence): end_pos]
            str2 = sent[end_pos: min(len(sent) - 1, end_pos + len(tokens_t[-1]) + 1)]
            str3 = str1 + str2
            str4 = str3.lstrip(' ')
            str5 = str4.rstrip(' ')
            str6 = str5 if not (str5.endswith(",") or str5.endswith(".") or str5.endswith(";")) else str5[:-1]
            noun_chunks_after_action_strs.append(str6)
        tam_passive_proactive = 0
        if action_chunk_verb.endswith("ed") or action_chunk_verb == "bound":
            t_a = chunk_tree[action_chunk_index + 1]
            t_a_tokens = [w[0] for w in t_a]
            if t_a_tokens[0] == "by":
                tam_passive_proactive += 1
        passive_chek = True if tam_passive_proactive > 0 else False
        passive_tobe = {"is": sent.find("is", 0, action_position_in_sentence),
                        "are": sent.find("are", 0, action_position_in_sentence),
                        "was": sent.find("was", 0, action_position_in_sentence),
                        "were": sent.find("were", 0, action_position_in_sentence)}
        max_pos = -1
        element = ["is"]
        for key in passive_tobe.keys():
            if passive_tobe[key] > max_pos:
                max_pos = passive_tobe[key]
                element.append(key)
        tobe_passive_verb = element[-1]
        passive_action_verb = tobe_passive_verb + " " + action_chunk_verb + " by"
        if passive_chek == False:
            list_all_verb_and_its_noun_chunks.append([chunk_tree, action_chunk_verb, noun_chunks_before_action_indexs, noun_chunks_before_action_strs, noun_chunks_after_action_indexs, noun_chunks_after_action_strs, passive_chek])
        else:
              list_all_verb_and_its_noun_chunks.append(
                [chunk_tree, passive_action_verb, noun_chunks_before_action_indexs, noun_chunks_before_action_strs,
                 noun_chunks_after_action_indexs, noun_chunks_after_action_strs, passive_chek])
    return list_all_verb_and_its_noun_chunks

# features_dictionary is a function that has input chunk_tree, index_noun1, action, index_noun2
# where chunk_tree is a tree of chunks, index_noun1 and index_noun2 are 2 indexes of 2 noun chunks
# in the tree, and action is a string of a verb
# the purpose of this function is that with a triple <noun1, verb, noun2>, the function would create a
# dictionary to extract good features of this triple, and return this dictionary
def features_dictionary(chunk_tree, index_noun1, action, index_noun2):   # action is a string of verb
    noun_chunk1_pairs= chunk_tree[index_noun1].leaves()  # a list of (word,tag)   #  error maybe here: AttributeError: 'tuple' object has no attribute 'leaves'
    # compute distance between 2 chunks
    distance = index_noun2-index_noun1
    ###
    #count numbers of noun chunks between 2 chunks
    count_noun_chunk_between =0
    for i in range(index_noun1+1, index_noun2):
        try:
            sub_chunk = chunk_tree[i]
            if sub_chunk.label()=="NP":
                count_noun_chunk_between+=1
        except:
            pass
    # if action is of the form Verbs, for example activates, then there should be NNS types in noun_chunk1
    noun_tags=[]
    singular_plural = 0
    if len(noun_tags)>0:
        for pair in noun_chunk1_pairs:
            if pair[1] == "NN" or pair[1] == "NNS":
                noun_tags.append(pair[1])
        if noun_tags[-1] == "NNS" and (not action.endswith("s")):
            singular_plural += 1
        if noun_tags[-1] == "NN":
            if action.endswith("ed") or action.endswith("s"):
                singular_plural += 1

    plural_satisfy = True if singular_plural>0 else False

    # list of pos_tag between 2 noun phrases
    list_pos_tag_between=[]
    for i in range(index_noun1+1,index_noun2):
        t= chunk_tree[i]
        pairs_t=chunk_tree.leaves()
        pos_t=[pair[1] for pair in pairs_t]
        list_pos_tag_between.extend(pos_t)
    tuple_pos_tag_between=tuple(list_pos_tag_between)
    list_distinct_pos_tag_between=list(set(list_pos_tag_between))

    # check so that this form would not happen: (NP promiscuous/JJ)
    check_noun1=0
    check_noun2=0
    try:
        for pair in chunk_tree[index_noun1].leaves():
            if pair[1] in ["NN", "NNS"]:
                check_noun1 += 1
    except:
        pass

    try:
        for pair in chunk_tree[index_noun2].leaves():
            if pair[1] in ["NN", "NNS"]:
                check_noun2 += 1
    except:
        pass

    check_containing_NN_pos= (check_noun1>0) and (check_noun2>0)   # it is True or False

    dict_feature={"distance": distance, "count_noun_chunk_between": count_noun_chunk_between, "plural_satisfy":plural_satisfy,"list_pos_tag_between": tuple_pos_tag_between,"check_NN_pos":check_containing_NN_pos}
    return dict_feature

# now from 100 sentences with index 0,1,... 99, we would choose 20 indexes to
# be indexes for test sentences, and the remaining 80 indexes would be indexes for
# train sentences
list_index_test_sentences=[0,4,6,7,20,27,29,30,40,41,43,44,61,63,68,73,83,86,87,91]
list_index_train_sentences=[i for i in list(range(100)) if i not in list_index_test_sentences]

# generate_train_data is a function that create a list of tuple (dictionary of features of triple, label of triple)
# to be used for training
def generate_train_data(): # list of strings, each string is a sentence
    train_sentences_x_action_y= [contents[i][:2] for i in list_index_train_sentences]
    train_sentences=[f[0] for f in train_sentences_x_action_y]  # list of strings
    # train_sentences_x_action_y is a list of (sentence,x,action,y)
    train_data=[]
    # action_chunk_verb, noun_chunks_before_action_indexs, noun_chunks_before_action_strs, noun_chunks_after_action_indexs, noun_chunks_after_action_strs, passive_chek
    for i in range(len(train_sentences)):
        all_verbs_its_noun_chunks = get_noun_chunk(train_sentences[i])  # a list of the form [[x1,x2,..,x8],[x1,...,x8]]...
        for j in range(len(all_verbs_its_noun_chunks)):
            [chunk_tree, action, noun_chunks_before_action_indexs, noun_chunks_before_action_strs, noun_chunks_after_action_indexs, noun_chunks_after_action_strs, passive_chek] = all_verbs_its_noun_chunks[j]
            # these indexs are indexs inside the chunk tree
            chosen_noun_chunks_before_action_indexs = noun_chunks_before_action_indexs[-2:]
            chosen_noun_chunks_before_action_strs = noun_chunks_before_action_strs[-2:]
            chosen_noun_chunks_after_action_indexs = noun_chunks_after_action_indexs[:2]
            chosen_noun_chunks_after_action_strs = noun_chunks_after_action_strs[:2]
            expected_X = train_sentences_x_action_y[i][1][0][0]
            expected_ACTION = train_sentences_x_action_y[i][1][0][1]
            expected_Y = train_sentences_x_action_y[i][1][0][2]
            # features_dictionary(chunk_tree, index_noun1, action, index_noun2)
            for h1 in range(len(chosen_noun_chunks_before_action_indexs)):
                for h2 in range(len(chosen_noun_chunks_after_action_indexs)):
                    noun_before_str = chosen_noun_chunks_before_action_strs[h1]
                    noun_after_str = chosen_noun_chunks_after_action_strs[h2]
                    dict_fea = features_dictionary(chunk_tree, chosen_noun_chunks_before_action_indexs[h1], action,
                                                   chosen_noun_chunks_after_action_indexs[h2])
                    if (noun_before_str == expected_X) and (noun_after_str == expected_Y):
                        train_data.append((dict_fea, 1))
                        train_data.append((dict_fea, 1))
                        train_data.append((dict_fea, 1))
                        train_data.append((dict_fea, 1))
                        train_data.append((dict_fea, 1))
                    else:
                        train_data.append((dict_fea, 0))

    return train_data

# train_set is the data for training:
train_set= generate_train_data()
# now, we create the classifier to train:
classifier= nltk.MaxentClassifier.train(train_set,algorithm="gis",max_iter=30)

# annotate_a_new_sentence would take sent as input, where sent is a string for a sentence
# it would create an empty list named good_triple, and then collect possible triples of
# the form <X,ACTION,Y> from that sentence and use the classifierthat we have trained to
# output label for each triple, if the output label is 1, then appending that triple
# to the list good_triple. Finally, this function would return good_triple as a list
# of triples from sent that predicted by our module
def annotate_a_new_sentence(sent): # sent is a string
    list_all_verbs_its_noun_chunks= get_noun_chunk(sent)
    good_triples = []
    for i in range(len(list_all_verbs_its_noun_chunks)):
        [chunk_tree, action, noun_chunks_before_action_indexs, noun_chunks_before_action_strs, noun_chunks_after_action_indexs, noun_chunks_after_action_strs, passive_chek] = list_all_verbs_its_noun_chunks[i]
        # these indexs are indexs inside the chunk tree
        chosen_noun_chunks_before_action_indexs = noun_chunks_before_action_indexs[-4:]
        chosen_noun_chunks_before_action_strs = noun_chunks_before_action_strs[-4:]
        chosen_noun_chunks_after_action_indexs = noun_chunks_after_action_indexs[:2]
        chosen_noun_chunks_after_action_strs = noun_chunks_after_action_strs[:2]

        for h1 in range(len(chosen_noun_chunks_before_action_indexs)):
            for h2 in range(len(chosen_noun_chunks_after_action_indexs)):
                noun_before_str = chosen_noun_chunks_before_action_strs[h1]
                noun_after_str = chosen_noun_chunks_after_action_strs[h2]
                dict_fea = features_dictionary(chunk_tree, chosen_noun_chunks_before_action_indexs[h1], action,
                                               chosen_noun_chunks_after_action_indexs[h2])
                label = classifier.classify(dict_fea)
                if label == 1:
                    good_triples.append((noun_before_str, action, noun_after_str))
    return good_triples

# evaluate_train_set access performance of our module by giving 3 scores precision, recall, F1score
# for the performance of our module on 80 train sentences
def evaluate_train_set():
    train_sentences_x_action_y = [contents[i][:2] for i in list_index_train_sentences]
    train_sentences = [csv_sentences[i] for i in list_index_train_sentences]
    total_number_of_relation = 0
    for i in range(len(train_sentences)):
        list_x_action_y = train_sentences_x_action_y[i][1]
        total_number_of_relation += len(list_x_action_y)
    all_relation = []
    for i in range(len(train_sentences_x_action_y)):
        all_relation.extend(train_sentences_x_action_y[i][1])
    predicted_triples = []
    for sen in train_sentences:
        triples = annotate_a_new_sentence(sen)
        predicted_triples.extend(triples)
    true_predictions = 0
    for triple in predicted_triples:
        if triple in all_relation:
            true_predictions += 1
    Precision = true_predictions / len(predicted_triples)
    Recall = true_predictions / total_number_of_relation
    F1Score = 2 / ((1 / Precision) + (1 / Recall))
    return Precision, Recall, F1Score

"""
now, we call the function evaluate_train_set to get 3 scores precision, recall_train,f1score_train 
for the performance of our module on 80 train sentences
"""
precision_train,recall_train,f1score_train= evaluate_train_set()
print("Scores for performance on 80 train sentences: ")
print("Precision score : ",precision_train)
print("Recall score: ",recall_train)
print("F1score :",f1score_train)

# evaluate_test_set functions access performance of our module by giving 3 scores precision, recall, F1score
# for the performance of our module on 80 train sentences
def evaluate_test_set():
    test_sentences_x_action_y = [contents[i][:2] for i in list_index_test_sentences]
    test_sentences= [csv_sentences[i] for i in list_index_test_sentences]
    total_number_of_relation= 0
    for i in range(len(test_sentences)):
        list_x_action_y = test_sentences[i][1]
        total_number_of_relation += len(list_x_action_y)
    all_relation=[]
    for i in range(len(test_sentences_x_action_y)):
        all_relation.extend(test_sentences_x_action_y[i][1])
    predicted_triples=[]
    for sen in test_sentences:
        triples=annotate_a_new_sentence(sen)
        predicted_triples.extend(triples)
    true_predictions=0
    for triple in predicted_triples:
        if triple in all_relation:
            true_predictions+=1
    Precision = true_predictions/len(predicted_triples)
    Recall = true_predictions/total_number_of_relation
    F1Score= 2/((1/Precision) + (1/Recall))
    return Precision,Recall, F1Score

"""
now, we call the function evaluate_test_set to get 3 scores precision, recall_train,f1score_train 
for the performance of our module on 20 test sentences
"""
precision_test,recall_test,f1score_test=evaluate_test_set()
print("Score for performance on 20 test sentences : ")
print("Precision score: ",precision_test)
print("Recall score: ",recall_test)
print("F1score: ",f1score_test)


