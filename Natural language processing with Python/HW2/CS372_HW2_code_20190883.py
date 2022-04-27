
"""
                                    Idea for the code:
There are 2 main issues to deal with: finding adverbs that used as intensifier and among these adverbs,
choosing ones that just modify to a restrictive group.
I. My idea for the first issue:
 Firstly, i would use the corpus wordnet to collect all possible intensifier.
My idea is : in the wordnet corpus in nltk, each synset contains a group of words that have the same
meaning and type, and the common meaning for these words can be gotten through synset.definition(),
a call that return a string of definition. I create a list of about 20 intensifiers that I know and collect
the definitions of them into a list of string, and then separate these strings into separate words, then use
nltk.FreqDist to find the most common words in these strings. And the result is that: the word ‘intensifier’ ,
’intensifiers’, ‘degree’ is the most common words in the definition part, for example in wordnet, definition
for the Synset(‘very.r.01’) is  “used as intensifiers; `real' is …” ; and defition for Synset(‘highly.r.01’)
is ‘to a high degree or extent;…’. However, one important thing that I realize is: if the definition string
contains ‘intensifier’ or ‘intensifiers’, then it must be an intensifier, but if the definition string contains
‘degree’ then it can be an intensifier or a manner adverb. I create an empty list choose_intensifer=[ ] to
collect intensifiers and then go through all synsets in the wordnet, if the definition part of that synset
contains ‘degree’ or ‘intensifier’ or ‘intensifiers’ then append all synonyms in that synset to the list
choose_intensifier. As mentioned above, this list contains intensifiers but also maybe some manner adverbs,
so I have to score these adverbs inside list choose_intensifier. The score is as follow: with each adverb K,
initializing score_k=0, I access all adverb synset in the list nltk.Synsets(K) and if there is at least one
synset that its definition contains ‘intensifier’ or ‘intensifiers’ then return 1 for adverb K, otherwise
score_k = ( number of adverb synset that contains ‘degree’)/(number of adverb synsets from nltk.Synsets(K) ).
So the range of score_k is [0,1] and higher score_k mean higher possibility that K is an intensifier.

II. My idea for the second issue:
After collect all possible intensifiers from wordnet, I continue to deal
with the second issue: choosing adverbs that just modify to a restrictive group of adjectives.
And i deal with it as follow: using some text corpus to create a list of words as l1
(for example l1 = brown.words ), then creating a list l1_bigrams containing bigrams for l1,
and use filter function to get bigrams of  the form (adverb, adjective) that occur at least
twice in l1_bigrams. Next, I create an empty list A and among bigrams of the form (adv,adj)
in l1_bigrams, if adv is contained in choose_intensifier list ( defined above ) then the
bigram (adv,adj) would be appended to A. By this way, I have A as a list of (adverb,adjective)
bigrams that adverb is an intensifier. Next, for the issue of restrictiveness, I set a number
limit = 8 to be a threshold for restrictiveness, I create an empty list chosen_couple, and
among bigrams in list A, if the adverb in that bigram occurs less than limit times ( here, it is 8 times )
in A, then that bigram would be appended to the list chosen_couple. So, finally I have chosen_couple
is a list of bigrams of (adverb,adjective) that adverb is an intensifier and it modifies for just
a restrictive number of adjectives( less than 8 adjectives).

III. The final step is scoring, I give the formula for scoring a couple (D,E):
     Total_score = 2*intensifier_score + restrictive_score + 0.2*occurence_score       (a)
Here, as I mentioned in graph I, i gives score for each adverb to measure the possibility
that it is an intensifier, and in this formula, intensifier_score measures the possibility
that D can be an intensifier. I believe that this score is very important because I do not
want couples like (M,N) where M is a manner adverb to have high score, so I give the weight 2.
Next, as mentioned in graph II, from the list chosen_couple, one adverb D can appear in
several bigrams with different adjectives, so in the formula (a), restrictive_score is the
score measuring restrictiveness of D, and it ranges from 0 to 1, if D occurs in less number
of bigram in chosen_couple, then it gain higher score for its restrictiveness, and because
the restrictiveness is our main focus, so I give it weight 1. Finally, among couples that
have the same score given by 2*intensifier_score + restrictive_score , I rank them by the
number of times these couples appear in the original text, if the couple occur more, then
it gain higher score, the score occurrence_score range from [0,1] and because occurrence
is not our focus, so I give it weight 0.2

"""

import nltk
from nltk.corpus import wordnet
from nltk.corpus import brown, reuters, gutenberg, inaugural, webtext
from nltk import word_tokenize

from nltk.corpus import stopwords
from nltk import FreqDist
from nltk.collocations import *

stop_words = set(stopwords.words('english'))


# function read_line to read a file
def read_line(path):
    file = open(path, 'r')
    lines = file.readlines()
    results = []
    for line in lines:
        results.append(line.strip())
    return results


# remove all stop words from a list of words
def preprocess_corpus(word_list):
    stop_words = set(stopwords.words('english'))
    results = []
    for w in word_list:
        w_new = w.lower()
        if w_new.isalnum():
            if not w_new in stop_words:
                results.append(w_new)
    return results


# The following code would be important code that show my idea
############

"""
Following code would be the part that deal with the first issue mentioned above in graph I : 
collecting possible intensifiers from wordnet corpus
"""

# filter_contains function checks whether input string contains 'intensifier','intensifiers','degree' and
# does not contains 'manner'
# if satisfies the condition, then return 1, else return -1
# this function would be used as filter for bigram later
def filter_contains(s):
    tokens = word_tokenize(s)
    intensifer_specific = ['degree', 'intensifier', 'intensifiers']  # modifier nen duoc luu y
    manner_specific = ['manner']
    if (intensifer_specific[0] in tokens or intensifer_specific[1] in tokens or intensifer_specific[2] in tokens) and \
            manner_specific[0] not in tokens:
        return 1
    return -1


# function filter_one_word check whether the input word is an intensifier or not, if
# it satisfies the condition to be an intensifier, then returns 1, else return -1
# as mentioned in the idea part above, by using wordnet, i would access all
# adverb synset of a word w, then if there is any synset that its definition part contains
# 'intensifier' or 'intensifiers' or 'degree', than  it is considered to be intensifier

def filter_one_word(w):
    synsets = wordnet.synsets(w, wordnet.ADV)
    be_intensifer = 0
    for synset in synsets:
        definition = synset.definition()
        if filter_contains(definition) == 1:
            be_intensifer += 1
    if be_intensifer > 0:
        return 1
    return -1


# get_all_intensifier function create an empty list choosen_adverbs,
# then go through all synsets in the wordnet corpus. Among them, if one synset that
# its definition contains 'intensifier' or 'intensifiers' or 'degree' than
# all synonyms in this synset would be added to list choosen_adverb
# then i create an empty list double_filter
# with words in the list choosen_adverb, we use function filter_one_word
# to know whether it satisfies to be an intensifier, if it satisfies then
# it would be added to list double_filter
# the final_list list removes stop words from double and would be returned
# now we have final_list as a list containing all possible intensifiers from wordnet
def get_all_intensifier():
    all_synsets = [synset for synset in list(wordnet.all_synsets('r'))]
    choosen_adverbs = []
    for synset in all_synsets:
        definition = synset.definition()
        if filter_contains(definition) == 1:
            choosen_adverbs += synset.lemma_names()
    double_filter = []
    for adverb in choosen_adverbs:
        if filter_one_word(adverb) == 1 and '_' not in adverb:
            double_filter.append(adverb)
    stop_words = set(stopwords.words('english'))
    final_list = [w for w in double_filter if not w in stop_words]
    return list(set(final_list))


# filter_contains_scoring function is similar to filter_contains function but the difference is
# it gives a score to a string s, by considering whether it contains 'degree' or 'intensifier' or
# 'intensifiers'
# if string s contains 'intensifier' or 'intensifiers' then it must be the definition string of an
# intensifier, so it would be given max score 1, but if it only contains 'degree', then it can be
# definition string of an intensifier or a manner adverb, so it only gains score 0.5
def filter_contains_scoring(s):
    tokens = word_tokenize(s)
    intensifer_specific = ['degree', 'intensifier', 'intensifiers']  # modifier nen duoc luu y
    manner_specific = ['manner']
    if intensifer_specific[1] in tokens or intensifer_specific[2] in tokens:
        return 1
    if manner_specific[0] in tokens:
        return 0
    if intensifer_specific[0] in tokens:
        return 0.5
    return 0


# filter_one_word_scoring is smilar to the function filter_one_word but the difference is:
# it gives score for the input word to measure the possibility that it can be an intensifier.
# the range of this score is from 0 to 1 .
def filter_one_word_scoring(w):  # for an adverb from the list of chosen adverbs
    synsets = wordnet.synsets(w, wordnet.ADV)
    if len(synsets) == 0:
        return 0
    be_intensifer_scoring = 0
    for synset in synsets:
        definition = synset.definition()
        if filter_contains_scoring(definition) == 1:
            return 1
        else:
            be_intensifer_scoring += filter_contains_scoring(definition)
    return be_intensifer_scoring / len(synsets)  # in the case there is no intensifier term in the definition string


# get_all_intensifier_scoring function is similar to function get_all_intensifier
# but the difference is: it not only returns a list of words but also score for each
# word to measure the possibility that this word can be an intensifier
def get_all_intensifier_scoring():
    choosen_adverbs = get_all_intensifier()
    double_filter = []
    for adverb in choosen_adverbs:
        score = filter_one_word_scoring(adverb)
        if score > 0 and '_' not in adverb:
            double_filter.append((score, adverb))
    stop_words = set(stopwords.words('english'))
    final_list = [w for w in double_filter if not w in stop_words]
    return final_list


# function combine_corpus take a list of corpus's word list as input and then add these word list into one list
# and then return list all_bigram that contains all bigrams for this list.
# combine_corpus function also returns the list combine_adv_adjs that contains all couples (adverb,adjective)
# from list all_bigrams.
# combine_corpus function also return scores for the couples in list combine_adv_adjs.
def combine_corpus(corpus_words_list):
    combine_adv_adjs = []
    scores = []
    all_bigram = []
    for wordlist in corpus_words_list:
        all_bi, adv_adjs_score, adv_adjs = adv_adjs_bigrams(wordlist, adv_filter)
        combine_adv_adjs += adv_adjs
        scores += adv_adjs_score
        all_bigram += all_bi
    return all_bigram, combine_adv_adjs, scores

"""
The following code would deal with the second issue that i have mentioned in
graph II above, that is the restrictiveness of (D,E). This time, we have had 
functions to collect intensifiers from wordnet, and now,a among these intensifiers,
we would choose ones that only modify to a restricted group of adjectives.
"""


# adv_filter function takes a bigram w as input and check, if w[0] is adverb
# and w[1] is adjective then return 1, else function return -1
# this function would be used as a filter for bigram collocation later
def adv_filter(w):
    type_adj = ['JJ']
    type_adverb = ['RB']
    first = w[0]  # adv
    second = w[1]  # adj
    if first[1] not in type_adverb or second[1] not in type_adj:
        return -1
    return 1


# function adv_adjs_bigrams takes the list of words word_list, then create bigram
# for this list and return all bigram of the form (adverb,adjective)
def adv_adjs_bigrams(word_list, filter):  # this is a list of word from raw text
    pos_tag = nltk.pos_tag(word_list)
    bigram_measures = nltk.collocations.BigramAssocMeasures()

    # Ngrams with 'creature' as a member
    creature_filter = lambda *w: filter(w) == -1

    finder = BigramCollocationFinder.from_words(
        pos_tag)

    # only bigrams that appear 3+ times
    finder.apply_freq_filter(2)
    # only bigrams that contain 'creature'
    finder.apply_ngram_filter(creature_filter)
    # return the 10 n-grams with the highest PMI
    # print (finder.nbest(bigram_measures.likelihood_ratio, 10))
    total_list = []
    for i in finder.score_ngrams(bigram_measures.likelihood_ratio):
        total_list.append(list(i))
    list_without_score = []
    for ele in total_list:
        first = ele[0][0][0]
        second = ele[0][1][0]
        list_without_score.append([first, second])
    all_bigrams = nltk.bigrams(word_list)
    return all_bigrams, total_list, list_without_score

""" The last one is the last function to return the result couples, and the score formula
for each couple ( mentioned in graph III above ) would be included in this function and 
this score formula would be used to rank all couples, higher score mean higher ranking.
"""

# get_couple_D_E is the most important function, it return list of couples (adverbs,adjectives) to be chosen as
# well as their score for uniqueness and in the list, it has been sorted decreasingly basing on the score of each couple
# for example the return of get_couple_D_E() function is of the form [(('utterly', 'destroyed'), 2.626136105430584), (('utterly', 'destroy'), 2.625636219041127),...]
# as mentioned in the idea part, it would use function adv_adjs_bigram to genderate all bigram of the form
# (adverb,adjective) and then, choose bigrams (adv,adj) that adv appear in the list intensifier_list
# and adv appear less then limit times in the total_bigram list
def get_couple_D_E():
    intensifier_list = get_all_intensifier()
    intensifier_score_list = sorted(
        get_all_intensifier_scoring())  # [(0.05, 'badly'), (0.05, 'right'), (0.07142857142857142, 'heavily'), (0.07142857142857142, 'lightly'),...]

    intensifier_dict_score = {}
    for ele in intensifier_score_list:
        intensifier_dict_score[ele[1]] = ele[0]
    # intensifier_dict_score= {'wondrously':1,'wondrous': 1,...}
    l1 = brown.words()
    l2 = reuters.words()
    l3 = webtext.words()
    l4 = inaugural.words()
    l5 = gutenberg.words()
    l1_process = preprocess_corpus(l1)
    l2_process = preprocess_corpus(l2)
    l3_process = preprocess_corpus(l3)
    l4_process = preprocess_corpus(l4)
    l5_process = preprocess_corpus(l5)

    total_bigram, adv_adjs, _ = combine_corpus([l1_process, l2_process, l3_process, l4_process, l5_process])

    adv_adjs_0 = [w[0] for w in adv_adjs]
    adv_adjs_1 = [w[1] for w in adv_adjs]
    fdist_0 = FreqDist(adv_adjs_0)

    distint_adv = list(set(adv_adjs_0))
    dict_adverb = {}
    for adverb in distint_adv:
        dict_adverb[adverb] = []
    for couple_adv_adj in adv_adjs:
        if couple_adv_adj[1] not in dict_adverb[couple_adv_adj[0]]:
            dict_adverb[couple_adv_adj[0]].append(couple_adv_adj[1])
    # dict_adverb has the key is different adverbs from adv_adjs and its value is adjectives following it in adv_adjs0

    # count occurence of bigram:
    count_occurence_bigram = {}
    distint_bigram_from_total = list(set(total_bigram))
    for bigram in distint_bigram_from_total:
        count_occurence_bigram[bigram] = 0
    for bigram in total_bigram:
        count_occurence_bigram[bigram] += 1
    # the key for this dict is bigrams and value of each key is the number of occurence of that bigram
    max_occurence_bigrams = 0
    for bigram in distint_bigram_from_total:
        max_occurence_bigrams = max(max_occurence_bigrams, count_occurence_bigram[bigram])

    # choose adverb with the limit of adj follow from adv_adjs list
    limit = 8

    adv_frequent = []
    for intensifier in intensifier_list:
        if intensifier in adv_adjs_0:
            adv_frequent.append([intensifier, fdist_0[intensifier]])
    # adv_frequent = [['yet', 29], ['wildly', 1], ['widely', 1], ['wholly', 5], ['well', 21],...]  ---> adverbs that appear in intensifer list

    chosen_adv_num_follow = []
    for ele in adv_frequent:
        if ele[1] <= limit:
            chosen_adv_num_follow.append(ele)
    # chosen_adv_num_follow = [ ['wildly', 1], ['widely', 1], ['wholly', 5],...]
    chosen_adverbs = [w[0] for w in chosen_adv_num_follow]
    # chosen_adverbs= ['wildly','widely','wholly'...]

    max_adv_frequent = 0
    for ele in chosen_adv_num_follow:
        max_adv_frequent = max(max_adv_frequent, ele[1])

    # we give score to couples of adverbs and adjectives now
    dict_score_couples = {}
    for adv in chosen_adverbs:
        for adj in dict_adverb[adv]:
            intensifier_score = intensifier_dict_score[adv]
            occurence_score = (count_occurence_bigram[(adv, adj)]) / max_occurence_bigrams
            restrictive_score = len(
                dict_adverb[adv]) / max_adv_frequent  # number of adjectives that follow this adverb in bigrams
            total_score = 2 * intensifier_score + restrictive_score + 0.2 * occurence_score
            dict_score_couples[(adv, adj)] = total_score

    sorted_dict_score_couples = sorted(dict_score_couples.items(), key=
    lambda kv: (kv[1], kv[0]),reverse=True)  # [(('utterly', 'destroyed'), 2.626136105430584), (('utterly', 'destroy'), 2.625636219041127),...]
    # sorted_dict_score_couples is the decreasing sort of dict_score_couples basing on the total score

    # result_couples would be a list of chosen couples, and inside this list,
    # couples are in descending order of their score
    result_couples = [c[0] for c in sorted_dict_score_couples]
    # and it is a list
    return result_couples

# get first initial 100 couples with highest score
# and these couples are in the descending rank ( it means first couple in the list has rank 1, second couple has rank 2...)
initial_100 = get_couple_D_E()[:100]
print(initial_100)
