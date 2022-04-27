

import nltk
from nltk.corpus import  cmudict,brown,wordnet,stopwords, reuters,inaugural,webtext
from nltk import word_tokenize
from nltk.wsd import lesk
from wiktionaryparser import WiktionaryParser
import string

stop_words = set(stopwords.words('english'))
parser = WiktionaryParser()
phone = nltk.corpus.cmudict.dict()
wnl = nltk.WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

"""
Firstly, the implementation would generate a list to collect heteronyms
My idea to do this:
After spending time for experimenting, I realize that most of words that have 
at least two meanings and at least two way of pronunciation are more likely to be heteronyms,
therefore, to find for heteronyms, I will find for the words that have at least two meanings
and at least two pronunciations. Firstly, i initiate an empty list named list_heter, then I
search for words in nltk.cmudict to find words that have at least two pronunciations and add 
them to the list collect_heter. There is one thing to notice, that cmudict is a little bit outdated, 
therefore there are some words with just one pronunciation, but appearing to have 2 or more pronunciations 
in cmudict. Thus, I have to write a filter function to avoid adding them to the list. Next, to have a
more accurate collection, I use another dictionary named Wiktionary, and in the implementation, 
I write a parser that can parse word content from Wiktionary to JSON format, and with this parser, 
we can extract pronunciations of a word from the Wiktionary.
Then, with each words from list list_heter , I will extract pronunciations of these words from Wiktionary 
and check if their number of pronunciations are at least two or not, if not then I would exclude them from 
list collect_heter. Now, we have a list of words that have at least 2 ways of pronunciation, 
and we can trust this list because we have filtered it through two dictionaries: cmudict and Wiktionary,
and furthermore, Wiktionary is an online updated dictionary, so contents extracting from it are reliable.
Next, to deal with meaning issue, we can use nltk.corpus.wordnet because with any word A, we can 
have B= nltk.corpus.wordnet.synsets(A), then B is a list of synsets, and the number of synsets in B is 
the number of meanings of word A. Therefore, with each word in list collect_heter, I will use wordnet 
to check if its number of meaning is at least two, and if its number of meanings is less than two, 
I would exclude this word from list list_heter. From now on, we have list list_heter containings words 
with at least two meanings and two pronunciations. I will consider this list as a list containing heteronyms for the next parts.

"""
# contain_relation and contain_relation2 are functions would be used in get_Wik_IPA function
# to get all IPA pronunciations of a word
def contain_relation(s1,list_string):
    check = 0
    for s2 in list_string :
        if s1 in s2:
            check+=1
        if s2 in s1:
            check+=1
    if check >0:
        return True
    else:
        return False
def contain_relation2(s1,list_string,c2):
    for i in range(len(s1)):
        if (i==0):
            s11= c2+s1[1:]
            if s11 in list_string:
                return True
        elif (i==len(s1)-1):
            s11= s1[:-1]+c2
            if s11 in list_string:
                return True
        else:
            s11= s1[:i]+ c2 + s1[i+1:]
            if s11 in list_string:
                return True
    return False

# get_Wik_IPA function takes input string for a word, and return a list of IPA pronunciations for this word
def get_Wik_IPA(w):
    word = parser.fetch(w)
    list_all_IPA = []
    for i in range(len(word)):
        pronun_info = word[i]['pronunciations']['text']
        count_pronun = 0
        for j in range(len(pronun_info)):
            if ("US" in pronun_info[j]) or ("American" in pronun_info[j]):
                count_pronun += 1
                last_index_IPA = pronun_info[j].rfind('IPA')
                if last_index_IPA != -1:
                    sub = pronun_info[j][last_index_IPA:]
                    div_sub = sub.split(" ")
                    ipa = div_sub[1]
                    first_bra = ipa.find("/")
                    second_bra = ipa.rfind("/")
                    ipa = ipa[first_bra + 1: second_bra]
                    ipa = ipa.replace("ˈ", "")
                    ipa = ipa.replace("ˌ", "")
                    ipa = ipa.replace(".", "")
                    ipa_cop = ipa.replace("ɚ", "ə")
                    ipa_cop2 = ipa.replace("ə", "ɚ")
                    ipa_cop3= ipa.replace("i","ɨ")
                    ipa_cop4= ipa.replace("ɨ","i")
                    ipa_cop5 = ipa.replace("ɔ", "ɑ")
                    ipa_cop6 = ipa.replace("ɑ", "ɔ")
                    ipa_cop7 = ipa.replace("æw", "aʊ")
                    ipa_cop8 = ipa.replace("ɪ", "ɛ")
                    ipa_cop9 = ipa.replace("i", "ɪ")
                    if (ipa_cop not in list_all_IPA) and (ipa_cop2 not in list_all_IPA) and (ipa_cop3 not in list_all_IPA) and (ipa_cop4 not in list_all_IPA) and (ipa_cop5 not in list_all_IPA) and (ipa_cop6 not in list_all_IPA) and (ipa_cop7 not in list_all_IPA) and (ipa_cop8 not in list_all_IPA) and (ipa_cop9 not in list_all_IPA) and (not contain_relation2(ipa, list_all_IPA, "ɛ")) and (not contain_relation2(ipa, list_all_IPA, "i")) and (not contain_relation(ipa,list_all_IPA)):  # surprise error
                        list_all_IPA.append(ipa)
        if count_pronun == 0:
            if len(pronun_info) > 0:
                last_index_IPA = pronun_info[0].rfind('IPA')
                if last_index_IPA != -1:
                    sub = pronun_info[0][last_index_IPA:]
                    div_sub = sub.split(" ")
                    ipa = div_sub[1]
                    first_bra = ipa.find("/")
                    second_bra = ipa.rfind("/")
                    ipa = ipa[first_bra + 1: second_bra]
                    ipa = ipa.replace("ˈ", "")
                    ipa = ipa.replace("ˌ", "")
                    ipa = ipa.replace(".", "")
                    ipa_cop = ipa.replace("ɚ","ə")
                    ipa_cop2= ipa.replace("ə","ɚ")
                    ipa_cop3 = ipa.replace("i", "ɨ")
                    ipa_cop4 = ipa.replace("ɨ", "i")
                    ipa_cop5=ipa.replace("ɔ","ɑ")
                    ipa_cop6= ipa.replace("ɑ","ɔ")
                    ipa_cop7=ipa.replace("æw","aʊ")
                    ipa_cop8= ipa.replace("ɪ","ɛ")
                    ipa_cop9=ipa.replace("i","ɪ")
                    if (ipa_cop not in list_all_IPA) and (ipa_cop2 not in list_all_IPA) and (ipa_cop3 not in list_all_IPA) and (ipa_cop4 not in list_all_IPA) and (ipa_cop5 not in list_all_IPA) and (ipa_cop6 not in list_all_IPA) and (ipa_cop7 not in list_all_IPA) and (ipa_cop8 not in list_all_IPA) and (ipa_cop9 not in list_all_IPA) and (not contain_relation2(ipa, list_all_IPA, "ɛ")) and (not contain_relation2(ipa,list_all_IPA,"i")) and (not contain_relation(ipa,list_all_IPA)):  # surprise error
                        list_all_IPA.append(ipa)
    return list(set(list_all_IPA))

# cmu_process takes as input the string for a word, and this function
# is used as a filter function .
def cmu_process(w):
    pronun= phone[w]
    if len(pronun)==2:
        pro0=pronun[0]
        pro1=pronun[1]
        str0= " ".join(pro0)
        str1= " ".join(pro1)
        if str0.replace("S", "R") == str1:
            return False
        if "R" in pro0:
            if str0.replace("R","S")==str1:
                return False
            if str0.replace("R","ER0")==str1:
                return False
        if str0.replace("ER0", "R") == str1:
            return False
        if str0.replace("AA1", "AO1") == str1:
            return False
        if str0.replace("AO1", "AA1") == str1:
            return False
        if str0.replace("OW0", "AH0") == str1:
            return False
        if "EY1" in pro0:
            if str0.replace("EY1","AA1") == str1:
                return False
        if str0.replace("AH0", "IH0") == str1:
            return False
        """if str0.replace("AH0", "UW0")==str1:
            return False"""
    return True

# check_heteronym is a function that takes as input the string for a word,
# and return True if the word has at least two meanings and two pronunciations,
# otherwise, it would return False, this function also return False for some words
# that ending with 'er' because many of these words are comparative form of adjective
# and return False for words ending with 's' because many of these words plural form
# of nouns
def check_heteronym(word):
    if (len(word) == 1) or (word[-1] == 's') or (word[-2:]=="er"):
        return False
    try:
        pronun_w = phone[word.lower()]
        meanings_w = wordnet.synsets(word)
        if (len(pronun_w) >= 2) and (len(meanings_w) >= 2) and cmu_process(word) :
            return True
    except Exception as e:
        return False
    return False

# get_heter function return a list of possible heteronyms that have been collected
def get_heter():
    a = cmudict.words()
    s = []
    for w in a:
        stem_w = wnl.lemmatize(w)
        s.append(stem_w)
    heter = []
    for w in s:
        if check_heteronym(w):
            heter.append(w)
    heter_tam=list(set(heter))
    result_tam = []
    for w in heter_tam:
        try:
            u = get_Wik_IPA(w)
            if len(u) >= 2:
                result_tam.append(w)
        except Exception as e:
            pass
    result_heter = [w for w in result_tam if w not in stop_words]

    return result_heter

list_heter=get_heter()

"""
Now this part would write functions to annotate pronunciation
My idea is :
I use Wiktionary because it has pronunciation information for vocabularies. The information page 
of Wiktionary for one vocabulary A is organized as following: one vocabulary can have many Etymologies, 
so the content would be divided according to etymologies of A and content is divided into the form 
Ety_1,Ety_2,…Ety_n where Ety_k correspond to kth  etymology of the vocabulary A. Each of Ety_1,Ety_2,… has 
a unique pronunciation for A, and pronunciations associated with Ety_1,Ety_2,… are different from each other. 
Furthermore, each of Ety_1,Ety_2,… contains a list of different definitions for A. Assuming these list of 
definitions are Def_1,Def_2,…Def_k , then they are list of definitions of A and Def1 is contained in Ety_1 part,Def_2 is
contained in Ety_2 part,… And basing on this structure of Wiktionary, my idea for annotating 
pronunciation is : Assume we have sentence X, we use nltk.wsd.lesk to get definition string of each heteronym 
inside X, assuming heteronym H has definition string K, then I write a function  to measure the semantic 
similarity of two strings and use this function to measure the semantic similarity of definition string K
and all definition strings contained in definition lists Def_1,Def_2,…Def_k. Assume L is a string contained 
in a list Def_n that gives highest score when measuring semantic similarity with string K, then it is highly 
true that the definition string of heteronym H is contained in Def_n, and we already know that Def_n is contained 
in part Ety_n so I use the pronunciation associated with Ety_n to annotate for heteronym H. In this way, we can annotate pronunciation for all heteronyms of a sentences.
"""
# find_all_occurence function take as input (sentence,word), where sentence is a list of words that representing
# for a sentence, and word is a string of word that be contained in sentence
# this function return a list of occurrences position of word inside sentence
def find_all_occurence(sentence,word):
    list_occurence=[]
    for i in range(len(sentence)):
        if sentence[i]==word or wnl.lemmatize(sentence[i])==word:
            list_occurence.append(i)
    return list_occurence

# give_meaning takes as input (sentence,occurence_list,word) where sentence is a list of word that representing
# for a sentence, and word is a string of word that be contained in sentence, occurence_list is a list of occurence
# position of word inside sentece. By using nltk.lesk, give_meaning function would return a list of meanings for
# each position of word inside sentence
def give_meaning(sentence, occurence_list, word): # a list of meaning for each occurence of that word in the sentence
    if len(occurence_list)>0:
        list_meaning = []
        if len(occurence_list)==1:
            synset= lesk(sentence,word)
            list_meaning.append(synset.definition())
        else:
            for i in range(len(occurence_list)):
                if i==0:
                    start=0
                    end= occurence_list[1]
                    sub_sentence= sentence[start:end]
                    synset = lesk(sub_sentence, word)
                    list_meaning.append(synset.definition())
                elif i== len(occurence_list)-1:
                    start = occurence_list[i-1]
                    end= len(occurence_list)
                    sub_sentence = sentence[start:end]
                    synset= lesk(sub_sentence,word)
                    list_meaning.append(synset.definition())
                else:
                    start= occurence_list[i-1]+1
                    end= occurence_list[i+1]
                    sub_sentence = sentence[start:end]
                    synset = lesk(sub_sentence, word)
                    list_meaning.append(synset.definition())
        return list_meaning
    else:
        return []

# dict_meaning take as input (sentence,list_words) where sentence is a list of words representing a sentence,
# and list_words is a list of words that be contained in sentence
# it return a dict where keys are words in list_words, and values are list of meanings of all occurrences of
# that word
def dict_meaning_info(sentence,list_words):  # return a dict, that keys are words from the list, and values are list of meanings of all occurences of that word
    real_list= list(set(list_words))
    dict_={}
    for w in real_list:
        dict_[w]=[]
    for w in real_list:
        all_occurrence= find_all_occurence(sentence,w)
        all_meanings= give_meaning(sentence,all_occurrence,w)
        dict_[w].extend(all_meanings)
    return dict_

# similarity_2_sentences takes as input two list of words, s1, s2
# that each of them represent a sentence
# this function return the meanings similarity of these two sentences
# it would be used for annotating pronunciation for a word in later functions
def similarity_2_sentences(s1,s2): # inputs are two tokens
    l1 = []
    l2 = []
    s1_set = {w for w in s1 if not w in stop_words}
    s2_set = {w for w in s2 if not w in stop_words}
    rvector = s2_set.union(s2_set)
    for w in rvector:
        if w in s1_set:
            l1.append(1)  # create a vector
        else:
            l1.append(0)
        if w in s2_set:
            l2.append(1)
        else:
            l2.append(0)
    c = 0
    for i in range(len(rvector)):
        c += l1[i] * l2[i]
    cosine = c / (1+float((sum(l1) * sum(l2)) ** 0.5))
    return cosine

# dict_wik_definition_pronun function takes as input a string of the word w,
# and it returns a list of dictionary, each dictionary correspond to an Etymology of word w,
# each dictionary has keys : ['verb','noun','adjective','adverb',"pronun"]  ,
# values corresponding to keys 'verb','noun','adjective','adverb' are list of definition strings for word w, according to
# each of these pos-tag, and these definition strings are extracted from the dictionary Wiktionary using a parser.
# value of the key 'pronun' is the IPA pronunciation of w corresponding to its Etymology
def dict_wik_definition_pronun(w): # return a list of dictionary, each dictionary has keys : ['verb','noun','adjective','adverb',"pronun"] and value for each key is the list of definitions for that pos-tag
    word = parser.fetch(w)
    list_dict=[]
    for i in range(len(word)):
        if len(word[i]['pronunciations']['text'])>0:
            definition_dict = {}
            Wikt_pos = ['verb', 'noun', 'adjective', 'adverb']
            for pos in Wikt_pos:
                definition_dict[pos] = []
            for j in range(len(word[i]['definitions'])):
                part_of_speech = word[i]['definitions'][j]['partOfSpeech']
                if part_of_speech in Wikt_pos:
                    definition_dict[part_of_speech].extend(word[i]['definitions'][j]['text'])
            # s= word[0]['pronunciations']['text'] # a list
            index = [0]
            for j in range(len(word[i]['pronunciations']['text'])):
                length = len(word[i]['pronunciations']['text'])
                if "American" in word[i]['pronunciations']['text'][length - 1 - j]:
                    index.append(length - 1 - j)
            for j in range(len(word[i]['pronunciations']['text'])):
                length = len(word[i]['pronunciations']['text'])
                if "US" in word[i]['pronunciations']['text'][length - 1 - j]:
                    index.append(length - 1 - j)
            dicide_index = index[-1]
            pronun_info = word[i]['pronunciations']['text']
            last_index_IPA = pronun_info[dicide_index].rfind('IPA')
            if last_index_IPA != -1:
                sub = pronun_info[dicide_index][last_index_IPA:]
                div_sub = sub.split(" ")
                ipa = div_sub[1]
                first_bra = ipa.find("/")
                second_bra = ipa.rfind("/")
                ipa = ipa[first_bra + 1: second_bra]
                definition_dict["pronun"] = ipa

            list_dict.append(definition_dict)

    return list_dict

# transfer_pos_tag_nltk_2_Wikt is a function that takes as input a pos-tag in nltk,
# and return the corresponding pos-tag in Wiktionary for each part-of-speech
def transfer_pos_tag_nltk_2_Wikt(pos): # output would be one of strings: ["noun","verb","adjective","coording conjunction","determinant","cardinal digit","preposition", "TO"
    if pos in ["JJ","JJR","JJS"]:
        return "adjective"
    if pos in ["NNS","NN","NNP","NNPS"]:
        return "noun"
    if pos in ["RB","RBR","RBS"]:
        return "adverb"
    if pos in ["VB","VBD","VBG","VBN","VBP","VBZ"]:
        return "verb"
def transfer_Wikt_2_wordnet_pos(pos):
    if pos=="nound" :
        return "n"
    if pos=="verb" :
        return "v"
    if pos=="adjective":
        return "a"
    if pos=="adverb":
        return "r"

# wordnet_meanings function takes as input the string for the word w, and
# its pos-tag
# this function returns the list of definition strings for the word w
# by using nltk.wordnet
def wordnet_meanings(w,pos):
    syns=wordnet.synsets(w,pos)
    collect_def=[]
    for syn in syns:
        collect_def.append(syn.definition())
    return collect_def

# Wikt_dict_definition_for_heter function takes as input a list of words
# and return a python dictionary that give information about pronunciation
# and meanings of a word
def Wikt_dict_definition_for_heter(list_heteronym):
    dict_definition={} # a dict that key is word and value is list
    for w in list_heteronym:
        dict_definition[w]=dict_wik_definition_pronun(w)
    return dict_definition

heter_pronunciation_IPA=Wikt_dict_definition_for_heter(list_heter)

# refer_pronunciation function takes as input (w,meaning,nltk_pos)
# where w is a string for the word w, meaning is a definition string of w
# that been extracted from wordnet
# nltk_pos is the pos-tag of w
# this function return the IPA pronunciation for the word w basing
# on these inputs
def refer_pronunciation(w,meaning,nltk_pos):
    Wikt_pos=transfer_pos_tag_nltk_2_Wikt(nltk_pos)
    index_max_score=[0]
    max_score=0
    dict_heter=heter_pronunciation_IPA[w]
    for i in range(len(dict_heter)):
        list_pos_meaning=dict_heter[i][Wikt_pos]
        for ele_meaning in list_pos_meaning:
            similarity_score= similarity_2_sentences(meaning,ele_meaning)
            if similarity_score>max_score:
                max_score=similarity_score
                index_max_score.append(i)
    return dict_heter[index_max_score[-1]]["pronun"]

# info_word function takes as input (sentence,list_words)
# where sentence is a list of words that represent a sentence and
#list_words is a list of words
# it returns a python dictionary that give informations about occurrence positions,
# pos-tag for each occurence, list of meanings according to each occurrence and list
# of IPA pronunciation according to each occurrence of each word of list_words inside the sentence
def info_word(sentence,list_words):
    info_dict={}
    list_postag=nltk.pos_tag(sentence)
    for word in list_words:
        info_dict[word]=[[],[],[],[]]   # occurence, pos_tag according, list_meaning according, list pronunciation according
        all_occurence=find_all_occurence(sentence,word)
        info_dict[word][0].extend(all_occurence)
        for j in all_occurence:
            info_dict[word][1].append(list_postag[j][1])
        list_meaning= give_meaning(sentence,all_occurence,word)
        info_dict[word][2].extend(list_meaning)
        list_pronun =  []
        for h in range(len(all_occurence)):
            pronun= refer_pronunciation(word,list_meaning[h],info_dict[word][1][h])
            list_pronun.append(pronun)
        info_dict[word][3].extend(list_pronun)
    return info_dict

# list_occurence_pronunciation_heter takes sentence as input, where sentence is a list
# of words representing a sentence, and it returns a list of occurrence positions of each heteronym
# if they appear in the sentence, as well as list of pronunciation corresponding to each occurrence
def list_occurence_pronunciation_heter(sentence): # input is a token, that is a list of words
    list_heter_inside=[]
    for w in sentence:
        if w in list_heter and w not in list_heter_inside:
            list_heter_inside.append(w)
    dict_info = info_word(sentence,list_heter_inside)
    list_all_occurence=[]
    list_all_pronunciation=[]
    for w in sorted(dict_info.keys()):   # info_dict[word]=[[],[],[],[]]   # occurence, pos_tag according, list_meaning according, list pronunciation according
        for position in dict_info[w][0]:
            list_all_occurence.append(position)
        for pronun in dict_info[w][3]:
            list_all_pronunciation.append(pronun)
    return list_all_occurence,list_all_pronunciation

# filter1 functions take as input a list of sentence and return a list
# of sentence that contains at least 2 heteronyms
def filter1(list_sentences):
    collect_sentence=[]
    for sent in list_sentences:
        valid_sentence=0
        for w in sent:
            if w in list_heter:
                valid_sentence+=1
        if valid_sentence>1:
            collect_sentence.append(sent)
    return collect_sentence

"""
Now would be the part of scoring and giving ranking for sentences
I use the following formula to give score for each sentence:
Score = 100*score_1_2 + score_3           (a)
where  score_1_2 is the score for priority 1 and 2, and it is the number of occurrences 
of homograph inside a sentence, then obviously, sentences containing more occurrences of 
homograph would have higher score than ones with fewer, and sentences with occurrences
of homograph would have higher score than sentences not containing homograph. Score_3
is the score for priority 3, and it is the number of heteronyms with same part-of-speech 
information inside the sentence, then obviously, sentences containing heteronyms with same 
part-of-speech would have higher score. To keep the priority (1)>(2)>(3), I give the weight 100 for
score_1_2 and weight 1 for score_3 in formula (a).

"""

# score_priority1_priority2_priority3 is a function that give scores for the input sentence
# basing on 3 priorities in the pdf guide file
def score_priority1_priority2_priority3(sentence):
    score12=0
    list_heter_inside_sentence=[]
    for w in sentence:
        if w in list_heter:
            if w not in list_heter_inside_sentence:
                list_heter_inside_sentence.append(w)

    dict_meaning= dict_meaning_info(sentence,list_heter_inside_sentence)
    num_homo=0
    for w in list_heter_inside_sentence:
        if len(find_all_occurence(sentence,w))>=2:
            num_homo+=1
    score12+=num_homo
    for w in list_heter_inside_sentence:
        score12+= len(dict_meaning[w])*0.1

    sentence_pos= nltk.pos_tag(sentence)
    num_word_one_part_of_speech=0
    for w in list_heter_inside_sentence:
        all_occurrence_w= find_all_occurence(sentence,w)
        list_pos_w=[]
        for position in all_occurrence_w:
            list_pos_w.append(sentence_pos[position])
        if len(list(set(list_pos_w)))==1 and len(all_occurrence_w)>=2:
            num_word_one_part_of_speech+=1
    score3= num_word_one_part_of_speech
    score = 100*score12 + score3

    return score

# scoring is a function that takes as input a list of sentence
# and returns a list of score for each sentence
def scoring(list_sentences): # each sentence has at least one heteronym
    score_list=[]
    for sent in list_sentences:
        score_sent = score_priority1_priority2_priority3(sent)
        score_list.append(score_sent)
    return score_list

# sort_wordlist_score_list is a function that have inputs sent_list, score_list
# where sent_list is a list of sentences, and score_list is a list of scores,
# and sentence at index i of sent_list correspond to the score at index i of score_list
# this function helps to sort sentences in sent_list and scores in score_list so that
# scores in score_list are in decreasing order
def sort_wordlist_score_list(sent_list,score_list):
    for i in range(len(score_list)):
        for j in range(i,len(score_list)):
            if score_list[i]<score_list[j]:
                tam= score_list[i]
                score_list[i]=score_list[j]
                score_list[j] = tam
                tam_sent=sent_list[i]
                sent_list[i]=sent_list[j]
                sent_list[j]=tam_sent

# select_sentence is a function that search for 4 corpuses Brown, Reuter, Webtext, Inaugural
# to collect sentences that contain at least two heteronyms, and append them to the list all_sentences
# at the same time, select_sentence function also collect scores corresponding to these sentences,
# and append them to the list all_scores
# this function returns 2 lists all_sentences, all_scores
def select_sentence():
    collect_rank1_brown = filter1(brown.sents())
    zip_collect_brown = []
    for i in range(len(collect_rank1_brown)):
        zip_collect_brown.append(("Brown",collect_rank1_brown[i]))
    score_list_brown = scoring(collect_rank1_brown)

    collect_rank1_reuter = filter1(reuters.sents())
    zip_collect_reuter = []
    for i in range(len(collect_rank1_reuter)):
        zip_collect_reuter.append(("Reuter", collect_rank1_reuter[i]))
    score_list_reuter= scoring(collect_rank1_reuter)

    collect_rank1_webtext = filter1(webtext.sents())
    zip_collect_webtext = []
    for i in range(len(collect_rank1_webtext)):
        zip_collect_webtext.append(("Webtext", collect_rank1_webtext[i]))
    score_list_webtext=scoring(collect_rank1_webtext)

    collect_rank1_inaugural = filter1(inaugural.sents())
    zip_collect_inaugural = []
    for i in range(len(collect_rank1_inaugural)):
        zip_collect_inaugural.append(("Inaugural", collect_rank1_inaugural[i]))
    score_list_inaugural=scoring(collect_rank1_inaugural)

    all_sentences_corpus=  zip_collect_brown + zip_collect_reuter + zip_collect_webtext + zip_collect_inaugural
    all_sentences = [tup[1] for tup in all_sentences_corpus]

    all_scores= score_list_brown + score_list_reuter + score_list_webtext + score_list_inaugural
    return all_sentences_corpus, all_sentences, all_scores

# output_to_csv is a function that save sentences, cited corpus, heteronyms inside each sentence as well as pronunciations
# for these heteronyms into a csv file
def output_to_csv(list_sents, path_save): # input is a list of tuple, each tuple is of form ('Webtext', ['"', 'Out', 'for', '1', '.', '0', '"', 'heading'])
    with open(path_save,'w',encoding = 'utf-8') as f:
        for tuples in list_sents:
             list_all_occurence,list_all_pronunciation= list_occurence_pronunciation_heter(tuples[1])
             string_sentence= " ".join(tuples[1])
             f.write(string_sentence + " " + tuples[0] + "   ")
             for j in range(len(list_all_occurence)):
                 position_w= list_all_occurence[j]
                 f.write( tuples[1][position_w] + " : " + list_all_pronunciation[j] + "   ")
             f.write("\n")

# now, we have all_sentences is a list containing all sentences
# that sentence contains at least 2 heteronyms and all_scores
# is a list of score for each of sentence in all_sentences list
all_sentences_corpus,  all_sentences,  all_scores=select_sentence()
cop_score= all_scores
sort_wordlist_score_list(all_sentences_corpus,cop_score)
# now we need to sort them so that sentences with highest rankings would be at
# small index of the list all_sentences
sort_wordlist_score_list(all_sentences_corpus,all_scores)


output_list=[]
for tup in all_sentences_corpus:
    corp= tup[0]
    sent= tup[1]
    new_sent=" ".join(sent)
    output_list.append(("Corpus: "+ corp, new_sent))

# Now we print the 30 sentences with highest ranking, with cited corpus,
# because each sentence is very long, so it is very hard to annotate directly to it,
# therefore, i annotate pronunciations for each heteronym into separate columns in the CSV file
# but with this output, it is only name of cited corpus and the sentence
print(output_list[:30])









