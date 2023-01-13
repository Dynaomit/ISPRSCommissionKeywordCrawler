import itertools
import math
import re

import matplotlib.style
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import httplib2
import matplotlib.pyplot as plt
import difflib
from collections import Counter
from Levenshtein import ratio as levenshtein_distance


def request_website(link_string):
    http = httplib2.Http()
    _, response = http.request(link_string)
    return response


def get_commission_links():
    links = []
    response = request_website("https://www.isprs.org/publications/annals.aspx")
    for link in BeautifulSoup(response, features="html.parser").find_all('a'):
        link_ref = link.get('href')
        link_name = link.text
        if link_ref is not None \
                and link_name is not None \
                and "www.isprs-ann-photogramm-remote-sens-spatial-inf-sci.net" in link_ref \
                and "Commission" in link_name \
                and "Technical" not in link_name:
            links.append(link_ref)
    return links


def get_index_links_from_commission_links(commission_links):
    index_links = []
    for commission_link in commission_links:
        response = request_website(commission_link)
        for sub_link in BeautifulSoup(response, features="html.parser").find_all('a'):
            link_name = sub_link.text
            if link_name is not None and "Keyword index" in link_name:
                index_links.append(sub_link.get('href'))
    return index_links


def find_year_in_string(text):
    processed_string = re.sub('\s+', ' ', text).strip().lstrip()
    matches = re.findall(r'\d{4}', processed_string)
    if len(matches) != 0:
        # test = match.group(1)
        return matches[0]
    else:
        return None


def get_keywords_and_counts_basic(index_links):
    keywords = dict()
    current_keyword = ""
    for index_link in index_links:
        response = request_website(index_link)
        soup = BeautifulSoup(response, features="html.parser")
        current_year = find_year_in_string(soup.text)
        mydivs = soup.find_all("div", {"class": "indexData"})
        for div in mydivs:
            for content in div.contents:
                if content.name == "span":
                    current_keyword = re.sub('\s+', ' ', content.text).strip().lstrip().lower()
                if content.name == "ul":
                    if current_keyword not in keywords:
                        keywords[current_keyword] = {current_year: 1}
                    else:
                        if current_year not in keywords[current_keyword]:
                            keywords[current_keyword][current_year] = 1
                        else:
                            keywords[current_keyword][current_year] += 1
    return keywords


def plot_keywords(keyword_dict, top_n):
    matplotlib.style.use('tableau-colorblind10')
    top = dict(sorted(keyword_dict.items(), key=lambda x: sum(x[1].values()), reverse=True)[:top_n])
    df = pd.DataFrame(top).transpose()
    df = df.iloc[:, ::-1]
    df.sort_index(axis=0, ascending=False)
    df.plot(kind='bar', stacked=True)
    plt.xticks(rotation=45, ha='right', rotation_mode='anchor', fontsize=4)
    plt.yticks(fontsize=7)
    # plt.yticks(new_list)
    plt.tight_layout()
    plt.savefig('Top{} Keywords'.format(top_n), dpi=500)
    plt.show()


def sync_counts(k1, k2):
    for key in k2.keys():
        if key not in k1.keys():
            k1[key] = k2[key]
        else:
            k1[key] += k2[key]
    return k1


def get_overlap(s1, s2):
    string_list_1 = s1.replace('-', ' ').split(' ')
    string_list_2 = s2.replace('-', ' ').split(' ')
    return list(set(string_list_1).intersection(string_list_2))


def analyze_keys(keyword_dict):
    important_keys = dict()
    import time
    for first_key, second_key in itertools.permutations(keyword_dict.keys(), 2):
        #start_calc = time.time()
        dist = levenshtein_distance(first_key, second_key)
        #end_calc = time.time()
        #start_append = time.time()
        if first_key != second_key and dist > 0.8:
            overlap = get_overlap(first_key, second_key)
            if overlap:
                overlap_string = ' '.join(overlap)
                if overlap_string in important_keys:
                    important_keys[overlap_string].append(first_key)
                    important_keys[overlap_string].append(second_key)
                else:
                    important_keys[overlap_string] = [first_key, second_key]
        #end_append = time.time()
        #print("Calc time = ", end_calc-start_calc, " Append time = ", end_append-start_append)
    for key in important_keys:
        important_keys[key] = list(dict.fromkeys(important_keys[key]))
    return important_keys


def process_duplicates(keyword_dict):
    important_keys = analyze_keys(keyword_dict)
    keys_to_delete = []
    for i, key in enumerate(keyword_dict.keys()):
        for j, other_key in enumerate(keyword_dict.keys()):
            dist = levenshtein_ratio_and_distance(key, other_key, ratio_calc=True)
            if i != j and dist > 0.8:
                overlap = get_overlap(key, other_key).strip()
                if overlap != key and overlap != other_key and overlap in keyword_dict.keys():
                    new_values = dict(
                        Counter(keyword_dict[key]) + Counter(keyword_dict[other_key]) + Counter(keyword_dict[overlap]))
                    keyword_dict[overlap] = new_values
                    keys_to_delete.append(key)
                    keys_to_delete.append(other_key)
                else:
                    keyword_dict[overlap] = dict(Counter(key) + Counter(other_key))
                    if overlap == key:
                        keys_to_delete.append(other_key)
                    else:
                        keys_to_delete.append(key)
    for key_to_delete in list(dict.fromkeys(keys_to_delete)):
        keyword_dict.pop(key_to_delete)
    return keyword_dict


def levenshtein_ratio_and_distance(s, t, ratio_calc=False):
    """ levenshtein_ratio_and_distance:
        Calculates levenshtein distance between two strings.
        If ratio_calc = True, the function computes the
        levenshtein distance ratio of similarity between two strings
        For all i and j, distance[i,j] will contain the Levenshtein
        distance between the first i characters of s and the
        first j characters of t
    """
    # Initialize matrix of zeros
    rows = len(s) + 1
    cols = len(t) + 1
    distance = np.zeros((rows, cols), dtype=int)

    # Populate matrix of zeros with the indeces of each character of both strings
    for i in range(1, rows):
        for k in range(1, cols):
            distance[i][0] = i
            distance[0][k] = k

    # Iterate over the matrix to compute the cost of deletions,insertions and/or substitutions
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row - 1] == t[col - 1]:
                cost = 0  # If the characters are the same in the two strings in a given position [i,j] then the cost is 0
            else:
                # In order to align the results with those of the Python Levenshtein package, if we choose to calculate the ratio
                # the cost of a substitution is 2. If we calculate just distance, then the cost of a substitution is 1.
                if ratio_calc == True:
                    cost = 2
                else:
                    cost = 1
            distance[row][col] = min(distance[row - 1][col] + 1,  # Cost of deletions
                                     distance[row][col - 1] + 1,  # Cost of insertions
                                     distance[row - 1][col - 1] + cost)  # Cost of substitutions
    if ratio_calc:
        # Computation of the Levenshtein Distance Ratio
        ratio = ((len(s) + len(t)) - distance[row][col]) / (len(s) + len(t))
        return ratio
    else:
        # print(distance) # Uncomment if you want to see the matrix showing how the algorithm computes the cost of deletions,
        # insertions and/or substitutions
        # This is the minimum number of edits needed to convert string a to string b
        return "The strings are {} edits away".format(distance[row][col])


def main():
    commission_links = get_commission_links()
    index_links = get_index_links_from_commission_links(commission_links)
    keyword_dict = get_keywords_and_counts_basic(index_links)
    no_duplicate_dict = process_duplicates(keyword_dict)
    print(len(keyword_dict.keys()))
    plot_keywords(keyword_dict, 40)


if __name__ == '__main__':
    main()
