
from collections import defaultdict
import csv
import nltk
from enum import Enum


class Gender(Enum):
    UNKNOWN = 0
    MASCULINE = 1
    FEMININE = 2


# Mapping of (lowercased) pronoun form to gender value. Note that reflexives
# are not included in GAP, so do not appear here.
PRONOUNS = {
    'she': Gender.FEMININE,
    'her': Gender.FEMININE,
    'hers': Gender.FEMININE,
    'he': Gender.MASCULINE,
    'his': Gender.MASCULINE,
    'him': Gender.MASCULINE,
}

# Fieldnames used in the gold dataset .tsv file.
GOLD_FIELDNAMES = [
    'ID', 'Text', 'Pronoun', 'Pronoun-offset', 'A', 'A-offset', 'A-coref', 'B',
    'B-offset', 'B-coref', 'URL'
]
# Fieldnames expected in system output .tsv files.
SYSTEM_FIELDNAMES = ['ID', 'A-coref', 'B-coref']

import pickle

"""
                                    The main idea for this homework:
The only issue in this homework is: from a given string of many sentences with one annotated pronoun P and two 
annotated people name A, B, how we can determine whether P,A are coreferential and whether P,B are coreferential. 
My idea for this issue is : from the data read from file gap-development.tsv , we will create a training set and 
create a classifier by nltk.MaxentClassifier to train this training set, after that, with each line U in file gap-test.tsv, 
we will use the classifier to determine whether each pair (P,A) and (P,B) of U is coreferential. We have 2 tasks: 
snippet-context and page-context. With the snippet-context, to create the training set, I initiate an empty list 
named train_data, then write program to read each line in file gap-development.tsv, after that, with each line X, 
we will read 8 information: ID, text, P (the pronoun) , A (the first name ) , A-coref ( the boolean value corresponding 
to whether (P,A) is coreferential), B ( the second name) , B-coref (the boolean value corresponding to whether 
(P,B) is coreferential), URL(the link to the article on Wikipedia). From these information, we will generate a 
dictionary of features for pair (P,A) as follow : initiate an empty dictionary dict_ = {} , the first feature is 
the couple of A and P, that means dict_[“noun_pronoun”] =  (A, P) . The second feature is the distance of A 
and P on the test, that means the number of words between A and P on the test. The idea behind this feature is that 
couple (name,pronoun) with shorter distance would be more likely to be coreferential. To have the next feature, we 
need to write program for chunking a sentence, to do this, I write a chunk grammar and then use nltk. RegexpParser 
to create a chunk parser. Then, we can use this parser for chunking sentences of the text (text is a data that I have 
defined above), and the result is chunk trees of sentences of the text. Now, we can determine the position of A and P 
on these trees, we assume that the 2 subtrees containing A and P are t1 and t2, then the next feature to add to the 
dictionary dict_ is the distance of A and P on these trees of chunks, that means the number of subtrees between t1 and 
t2. The next feature is to check whether A is a subject or an object in its sentence. I choose this feature because 
according to some linguistic researches about coreference that I found on the internet, entities in the subject position 
is more likely to be referred to than entities in the object position. Assuming that A is contained in sentence M, then 
to check whether A is subject of M or not, we can use a very simple idea: firstly, chunking the sentence M to have a tree 
T, then among subtrees of T, if we can find a NP phrase (noun phrase) that is the child of T and the sibling of at least 
one VP phrase (verb phrase), then this NP phrase has high probability to be the subject of sentence M. Therefore, we can 
check, if A is contained in a NP subtree that is the child of T and this NP subtree has at least one VP sibling subtree, 
then dict_[“check_subject”]= True, otherwise, dict_[“check_subject”] = False, and this is our fourth feature. The final 
feature is based on a simple idea: more times A appears in the text (the text data is defined above) , then more likely 
that A is coreferential to the pronoun P. Therefore, we count the number of occurrences of A inside the text, and this 
is our final feature. Then, we add the couple (dict_ , A-coref ) to the list train_data. Similarly, we create a dictionary 
of features for (B,P) in the same way, then let name it second_dict, and we add the couple (second_dict, B-coref) to 
train_data. By doing this for all lines of the file gap-development.tsv, we have created the train set to be used for 
training for the task snippet-context. With the remaining task, page-context, we will generate the training set in the 
same way, but we will add one more feature when we create the dictionary of feature. In more detail, with the line X 
that we have read from file gap-development.tsv, we get its URL and then we can use this link to get the title of its 
Wikipedia article, let assume that this title is the string Y. Then, we have 2 strings A and Y, we will check whether 
string A and Y have common substring, if they have, then dict_[“overlap_title] = True, otherwise, dict_[“overlap_title] = 
False  and this is the feature that we have append for task page-context. After generating training data, we use 
nltk.MaxentClassifier to create a classifier, and then we use the training data above to train this classifier. Then, 
the next step is to predict the coreference relations of each line in file gap-test.tsv. To do this, when we read each 
line Z of this file, with pair (P,A) of Z where P is pronoun and A is name, we will create 2 dictionaries of features of 
(P,A) in 2 manners snippet-context and page-context in the way described above, then we use these dictionaries as the 
input for the classifier that we have trained, and then the output of the classifier would be True or False. This is the 
prediction of our model about whether P,A is coreferential. We do in the same way for pair (P,B) of line Z. By doing this, 
we can predict the coreference relations of each line in file gap-test.tsv

"""

# the class Annotation, function read_annotations would help to read data from tsv file
class Annotation(object):
    """Container class for storing annotations of an example.

    Attributes:
      gender(None): The gender of the annotation. None indicates that gender was
        not determined for the given example.
      name_a_coref(None): bool reflecting whether Name A was recorded as
        coreferential with the target pronoun for this example. None indicates
        that no annotation was found for the given example.
      name_b_coref(None): bool reflecting whether Name B was recorded as
        coreferential with the target pronoun for this example. None indicates
        that no annotation was found for the given example.
    """

    def __init__(self):
        self.gender = None
        self.name_a_coref = None
        self.name_b_coref = None
        self.A = None
        self.B = None
        self.pronoun = None
        self.text = None
        self.url = None


def read_annotations(filename, is_gold):
    """Reads coreference annotations for the examples in the given file.

    Args:
      filename: Path to .tsv file to read.
      is_gold: Whether or not we are reading the gold annotations.

    Returns:
      A dict mapping example ID strings to their Annotation representation. If
      reading gold, 'Pronoun' field is used to determine gender.
    """

    def is_true(value):
        if value.lower() == 'true':
            return True
        elif value.lower() == 'false':
            return False
        else:
            print('Unexpected label!', value)
            return None

    fieldnames = GOLD_FIELDNAMES if is_gold else SYSTEM_FIELDNAMES
    annotations = defaultdict(Annotation)
    with open(filename, 'rU') as f:
        reader = csv.DictReader(f, fieldnames=fieldnames, delimiter='\t')

        # Skip the header line in the gold data
        if is_gold:
            next(reader, None)

        for row in reader:
            example_id = row['ID']
            if example_id in annotations:
                print('Multiple annotations for', example_id)
                continue
            annotations[example_id].pronoun = row["Pronoun"]
            annotations[example_id].A = row['A']
            annotations[example_id].name_a_coref = is_true(row['A-coref'])
            annotations[example_id].B = row['B']
            annotations[example_id].name_b_coref = is_true(row['B-coref'])
            annotations[example_id].text = row["Text"]
            annotations[example_id].url = row["URL"]
    return annotations

# function get_dict_from_gold takes string filename as input, and then it read informations from that
# tsv file such as example_id, pronoun, A, name_a_coref, B, name_b_coref, text, url ,
# and this function return a dictionary with keys are example_id of lines of the file, and value are
# information for each line such as : pronou, A, B,...
def get_dict_from_gold(filename):
    file = read_annotations(filename, True)
    dict_ = {}
    for example_id, gold_annotation in file.items():
        dict_[example_id] = {}
    for example_id, gold_annotation in file.items():
        dict_[example_id]["example_id"] = example_id
        dict_[example_id]["pronoun"] = gold_annotation.pronoun
        dict_[example_id]["A"] = gold_annotation.A
        dict_[example_id]["name_a_coref"] = gold_annotation.name_a_coref
        dict_[example_id]["B"] = gold_annotation.B
        dict_[example_id]["name_b_coref"] = gold_annotation.name_b_coref
        dict_[example_id]["text"] = gold_annotation.text
        dict_[example_id]["url"] = gold_annotation.url
    return dict_

# write_to_tsv function take list_to_write and filename as input where list_to_write is a list of list of element that
# we need to write to the tsv file with name filename
def write_to_tsv(list_to_write,filename):  # a list of list, for example: [['test-1',True,False], ['test-2', True, True],... ]
    with open(filename, 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for info_list in list_to_write:
            tsv_writer.writerow(info_list)

# de-non_alpha is a function to process string
def de_non_alpha(str):
    str1 = str.lstrip()
    str2 = str.rstrip()
    new_str = ""
    for i in range(len(str2) - 1):
        new_str += str2[i]
    if str2[-1] != ',' and str2[-1] != "'":
        new_str += str2[-1]
    return new_str


# list_position_inside_text takes str and text as input where str is string of several words, and text is a string of
# several sentences, this function return the position of str inside text
def list_position_inside_text(str, text):
    words_of_str = str.split(" ")
    head_word = de_non_alpha(words_of_str[0])
    tam_list_tokens = nltk.word_tokenize(text)
    list_tokens = []
    for i in range(len(tam_list_tokens)):
        if head_word in tam_list_tokens[i]:
            list_tokens.append(head_word)
        else:
            list_tokens.append(tam_list_tokens[i])
    list_token_position = []
    for i in range(len(list_tokens)):
        if list_tokens[i] == head_word:
            list_token_position.append(i)
    list_sents = nltk.sent_tokenize(text)  # a list of sentence
    list_sentence_position = []
    for i in range(len(list_sents)):
        if str in list_sents[i]:
            list_sentence_position.append(i)
    if len(list_sentence_position) == 0:
        for i in range(len(list_sents)):
            if head_word in list_sents[i]:
                list_sentence_position.append(i)
    return list(sorted(list_token_position)), list(sorted(list_sentence_position))


# chunk_a_text take text as input where text is a string of several sentences
# this function return a list of chunk trees for all sentences of text
def chunk_a_text(text):
    list_sents = nltk.sent_tokenize(text)
    list_sentences_tokens = []
    for sent in list_sents:
        tokens = nltk.word_tokenize(sent)
        list_sentences_tokens.append(tokens)
    pos_text = nltk.pos_tag_sents(list_sentences_tokens)  # list(list(tuple))

    grammar = r"""
              NP: {<DT|JJ|NN.*>+}          # Chunk sequences of DT, JJ, NN
              PP: {<IN><NP>}               # Chunk prepositions followed by NP
              VP: {<VB.*><NP|PP|CLAUSE>+$} # Chunk verbs and their arguments
              CLAUSE: {<NP><VP>}           # Chunk NP, VP
              """
    cp = nltk.RegexpParser(grammar)
    text_chunk_tree = []

    for sentence_pos in pos_text:
        chunk_single_sentence = cp.parse(sentence_pos)
        text_chunk_tree.append(chunk_single_sentence)
    return text_chunk_tree

# distance_in_chunk_tree_of_a_text function takes  text, str0, sent_position0, str1, sent_position1
# as input where text is the string of several sentences, str0 and str1 are strings of several words,
# sent_position0 and sent_position1 are the indexs of sentences of text that contain str0 and str1
# this function create chunk trees for all sentences of text, then determine the subtrees t1,t2 that
# contain str0 and str1, then this function will return the number of subtrees between t1 and t2
# so, the purpose of this function is to determine the distance of str0 and str1 on chunk trees
def distance_in_chunk_tree_of_a_text(text, str0, sent_position0, str1, sent_position1):
    list_sents = nltk.sent_tokenize(text)
    text_chunk = chunk_a_text(text)
    if (str0 in list_sents[sent_position0]) and (str1 in list_sents[sent_position1]):
        tree_chunk0 = text_chunk[sent_position0]
        list_position_str0_in_chunk0 = []
        for i in range(len(tree_chunk0)):
            try:
                leaves_tree = tree_chunk0[i].leaves()
                leaves_str = [w[0] for w in leaves_tree]
                if str0 in leaves_str:
                    list_position_str0_in_chunk0.append(i)
            except:
                pass
        if len(list_position_str0_in_chunk0) == 0:
            list_position_str0_in_chunk0.append(0)

        tree_chunk1 = text_chunk[sent_position1]
        list_position_str1_in_chunk1 = []
        for i in range(len(tree_chunk1)):
            try:
                leaves_tree = tree_chunk1[i].leaves()
                leaves_str = [w[0] for w in leaves_tree]
                if str1 in leaves_str:
                    list_position_str1_in_chunk1.append(i)
            except:
                pass
        if len(list_position_str1_in_chunk1) == 0:
            list_position_str1_in_chunk1.append(0)

        first_distance = abs(len(tree_chunk0) - list_position_str0_in_chunk0[0] + list_position_str1_in_chunk1[-1]) if (
                    sent_position0 != sent_position1) else abs(
            - list_position_str0_in_chunk0[0] + list_position_str1_in_chunk1[-1])

        num_chunk_of_distance_between = 0
        for index in range(sent_position0 + 1, sent_position1):
            chunk_tam = text_chunk[index]
            num_chunk_of_distance_between += len(chunk_tam)
        result_distance = num_chunk_of_distance_between + first_distance
        return result_distance
    else:
        return 0

# check_subject function takes tree,str as input where tree is a chunk tree and str is a string of
# several words, this function return a boolean value to indicate whether str is a subject of tree or not
def check_subject(tree, str):
    list_leaves_str = [w[0] for w in tree.leaves()]
    if str not in list_leaves_str:
        return False
    else:
        tam = 0
        try:
            list_subtree_labels = []
            for t in tree:
                list_subtree_labels.append(t.label())
            if ("VP" in list_subtree_labels) and ("NP" in list_subtree_labels):
                for t in tree:
                    t_leaves_str = [w[0] for w in t.leaves()]
                    if t.height() == 2 and (str in t_leaves_str):
                        if t.label() == "NP":
                            tam += 1
                    elif t.height() >= 3 and (str in t_leaves_str):
                        if t.label() == "NP":
                            for small_t in t:
                                try:
                                    tam_bo = small_t.height()
                                except Exception as e:
                                    if str == small_t[0]:
                                        tam += 1
                    else:
                        pass
        except Exception as e:
            pass
        if tam == 0:
            return False
        else:
            return True

# we have 2 tasks : snippet-context and page-context, now we will generate train set and classifier
# for task snippet-context first

# to create training set, with each pair of (P,A) and (P,B) in each line of file
# gap-development.tsv , we need to create a dictionary of features for (P,A) and (P,B)
# snippet_dict_feature_pronoun_A_B  is a function that help to generate dictionaries of features
# that would be used for training later
def snippet_dict_feature_pronoun_A_B(dictionary):
    pronoun = dictionary["pronoun"]
    text = dictionary["text"]
    A = dictionary["A"]
    B = dictionary["B"]

    # token distance
    A_list_token_position, A_list_sentence_position = list_position_inside_text(A, text)
    B_list_token_position, B_list_sentence_position = list_position_inside_text(B, text)
    pronoun_list_token_position, pronoun_list_sentence_position = list_position_inside_text(pronoun, text)

    A_token_position = A_list_token_position[0]
    A_sentence_position = A_list_sentence_position[0]

    B_tam_token_list = []
    for token_pos in B_list_token_position:
        if token_pos > A_token_position:
            B_tam_token_list.append(token_pos)
    B_token_position = 0
    if len(B_tam_token_list) > 0:
        B_token_position += B_tam_token_list[0]
    elif len(B_list_token_position) > 0:
        B_token_position += B_list_token_position[0]
    else:
        B_token_position += 0

    B_tam_sentence_list = []
    for sent_pos in B_list_sentence_position:
        if sent_pos >= A_sentence_position:
            B_tam_sentence_list.append(sent_pos)

    B_sentence_position = 0
    if len(B_tam_sentence_list) > 0:
        B_sentence_position += B_tam_sentence_list[0]
    elif len(B_list_sentence_position) > 0:
        B_sentence_position += B_list_sentence_position[0]
    else:
        B_sentence_position += 0

    pronoun_tam_token_list = []
    for token_pos in pronoun_list_token_position:
        if token_pos > A_token_position:
            pronoun_tam_token_list.append(token_pos)
    pronoun_token_position = 0
    if len(pronoun_tam_token_list) > 0:
        pronoun_token_position += pronoun_tam_token_list[0]
    elif len(pronoun_list_token_position) > 0:
        pronoun_token_position += pronoun_list_token_position[0]
    else:
        pronoun_token_position += 0

    pronoun_tam_sentence_list = []
    for sent_pos in pronoun_list_sentence_position:
        if sent_pos >= A_sentence_position:
            pronoun_tam_sentence_list.append(sent_pos)
    pronoun_sentence_position = 0
    if len(pronoun_tam_sentence_list) > 0:
        pronoun_sentence_position += pronoun_tam_sentence_list[0]
    elif len(pronoun_list_sentence_position) > 0:
        pronoun_sentence_position += pronoun_list_sentence_position[0]
    else:
        pronoun_sentence_position += 0

    # topical entity
    A_frequency = len(A_list_token_position)
    B_frequency = len(B_list_token_position)

    # syntactic distance
    head_A = A.split(" ")[0]
    head_B = B.split(" ")[0]
    distance_on_chunk_A_pronoun = distance_in_chunk_tree_of_a_text(text, head_A, A_sentence_position, pronoun,
                                                                   pronoun_sentence_position)
    distance_on_chunk_B_pronoun = distance_in_chunk_tree_of_a_text(text, head_B, B_sentence_position, pronoun,
                                                                   pronoun_sentence_position)

    # check subject
    text_chunk = chunk_a_text(text)
    check_A_subject = check_subject(text_chunk[A_sentence_position], head_A)
    check_B_subject = check_subject(text_chunk[B_sentence_position], head_B)

    feature_dict_pronoun_A = {}
    feature_dict_pronoun_A["noun_pronoun"] = (A, pronoun)
    feature_dict_pronoun_A["check_subject"] = check_A_subject
    feature_dict_pronoun_A["token_distance"] = abs(A_token_position - pronoun_token_position) // 15
    feature_dict_pronoun_A["frequency_noun"] = A_frequency
    feature_dict_pronoun_A["chunk_tree_distance"] = distance_on_chunk_A_pronoun // 8

    feature_dict_pronoun_B = {}
    feature_dict_pronoun_B["noun_pronoun"] = (B, pronoun)
    feature_dict_pronoun_B["check_subject"] = check_B_subject
    feature_dict_pronoun_B["token_distance"] = abs(B_token_position - pronoun_token_position) // 15
    feature_dict_pronoun_B["frequency_noun"] = B_frequency
    feature_dict_pronoun_B["chunk_tree_distance"] = distance_on_chunk_B_pronoun // 8

    return feature_dict_pronoun_A, feature_dict_pronoun_B

# generate_snippet_trainset function initiate an empty list named train_data
# and then with each (P,A) of each line from gap-development.tsv , this function
# create a dictionary for (P,A), let named it dict_ and then it add the
# couple (dict_, coref-A) to the list train_data. Finally, this function return the
# list train_data that would be used for training later
def generate_snippet_trainset():
    dict_ = get_dict_from_gold("gap-development.tsv")
    train_data = []
    for key in dict_.keys():
        info_dict = dict_[key]
        feature_dict_A_pronoun, feature_dict_B_pronoun = snippet_dict_feature_pronoun_A_B(info_dict)
        train_data.append((feature_dict_A_pronoun, info_dict["name_a_coref"]))
        train_data.append((feature_dict_B_pronoun, info_dict["name_b_coref"]))
    return train_data

# we generate training data and create the classifier
# and then use this training data to train the classifier
snippet_train_set = generate_snippet_trainset()
snippet_classifier = nltk.MaxentClassifier.train(snippet_train_set, algorithm="gis", max_iter=70)

# predict_snippet_coreference function will use the classifier that we have trained to
# predict the correference relations of each line in the file gap-test.tsv ,
# and then save these predictions to a list named predicted_list
def predict_snippet_coreference():
    dict_ = get_dict_from_gold("gap-test.tsv")
    keys = list(dict_.keys())
    predicted_list = []
    for key in keys:
        dictionary = dict_[key]
        dict_pronoun_A, dict_pronoun_B = snippet_dict_feature_pronoun_A_B(dictionary)
        prediction_A = snippet_classifier.classify(dict_pronoun_A)
        prediction_B = snippet_classifier.classify(dict_pronoun_B)
        predicted_list.append([dictionary["example_id"], prediction_A, prediction_B])
    return predicted_list

# we save the prediction of our model to file tsv
snippet_predicted_list = predict_snippet_coreference()
write_to_tsv(snippet_predicted_list, "CS372_HW5_snippet_output_20190883.tsv")

# now, we have done snippet-context task, we would move on to the page-context task

# longestSubstringFinder function would return the longest common substring of string1 and string2
def longestSubstringFinder(string1, string2):
    answer = ""
    len1, len2 = len(string1), len(string2)
    for i in range(len1):
        match = ""
        for j in range(len2):
            if (i + j < len1 and string1[i + j] == string2[j]):
                match += string2[j]
            else:
                if (len(match) > len(answer)): answer = match
                match = ""
    return answer

# page_context_dict_feature_pronoun_A_B function does the same job as function snippet_dict_feature_pronoun_A_B
# above but this time, this function does the job for task page-context
def page_context_dict_feature_pronoun_A_B(dictionary):
    feature_dict_pronoun_A, feature_dict_pronoun_B = snippet_dict_feature_pronoun_A_B(dictionary)
    url = dictionary["url"]
    A = dictionary["A"]
    B = dictionary["B"]
    title = url.split("/")[-1]
    common_substring_A_title = longestSubstringFinder(A, title)
    common_substring_B_title = longestSubstringFinder(B, title)
    check_overlap_A_title = True if len(common_substring_A_title) > 0 else False
    check_overlap_B_title = True if len(common_substring_B_title) > 0 else False
    feature_dict_pronoun_A["overlap_title"] = check_overlap_A_title
    feature_dict_pronoun_B["overlap_title"] = check_overlap_B_title
    return feature_dict_pronoun_A, feature_dict_pronoun_B

# generate_page_context_trainset function does the same job as function generate_snippet_trainset
# as above, but this time, this function does the job for task page-context
def generate_page_context_trainset():
    dict_ = get_dict_from_gold("gap-development.tsv")
    train_data = []
    for key in dict_.keys():
        info_dict = dict_[key]
        feature_dict_A_pronoun, feature_dict_B_pronoun = page_context_dict_feature_pronoun_A_B(info_dict)
        train_data.append((feature_dict_A_pronoun, info_dict["name_a_coref"]))
        train_data.append((feature_dict_B_pronoun, info_dict["name_b_coref"]))
    return train_data

# we generate train set for task page-context, and create a classifier
# and then use this train set to train the classifier
page_context_train_set = generate_page_context_trainset()
page_context_classifier = nltk.MaxentClassifier.train(page_context_train_set, algorithm="gis", max_iter=70)

# predict_page_context_coreference does the same job as the function predict_snippet_coreference
# above, but this time, for the page-context task
def predict_page_context_coreference():
    dict_ = get_dict_from_gold("gap-test.tsv")
    keys = list(dict_.keys())
    predicted_list = []
    for key in keys:
        dictionary = dict_[key]
        url = dictionary["url"]
        dict_pronoun_A, dict_pronoun_B = page_context_dict_feature_pronoun_A_B(dictionary)
        prediction_A = page_context_classifier.classify(dict_pronoun_A)
        prediction_B = page_context_classifier.classify(dict_pronoun_B)
        list_to_append = [dictionary["example_id"], prediction_A, prediction_B]
        predicted_list.append(list_to_append)
    return predicted_list

# we save our prediction for task page-context to the tsv file
page_context_predicted_list = predict_page_context_coreference()

write_to_tsv(page_context_predicted_list, "CS372_HW5_page_output_20190883.tsv")


