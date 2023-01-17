import re

import matplotlib.style
import pandas as pd
from bs4 import BeautifulSoup
import httplib2
import matplotlib.pyplot as plt
from collections import Counter
from Levenshtein import ratio as levenshtein_distance
import plotly.express as px


def request_website(link_string):
    http = httplib2.Http()
    _, response = http.request(link_string)
    return response


def get_commission_links():
    """
    searches the isprs annals website for commissions and collects their links
    :return: list of links to commissions
    """
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
    """
    from a list of links, which refer to commissions, parse the given websites for the link to each commission's keyword index
    :param commission_links: list of links to commissions
    :return: list of links to keyword indexes
    """
    index_links = []
    for commission_link in commission_links:
        response = request_website(commission_link)
        for sub_link in BeautifulSoup(response, features="html.parser").find_all('a'):
            link_name = sub_link.text
            if link_name is not None and "Keyword index" in link_name:
                index_links.append(sub_link.get('href'))
    return index_links


def find_year_in_string(text):
    """
    for a given text, represented by a string, return the first found year mentioned in said string
    :param text: large string to be searched
    :return: first found year, None if no year is found
    """
    processed_string = re.sub('\s+', ' ', text).strip().lstrip()
    matches = re.findall(r'\d{4}', processed_string)
    if len(matches) != 0:
        # test = match.group(1)
        return matches[0]
    else:
        return None


def get_keywords_and_counts_basic(index_links):
    """
    from a given list of links referring to keyword indexes,
    parses the keywords and counts how many papers mention each keyword
    the keywords are collected in a dictionary as keys to prevent duplicates
    :param index_links: given list of links to keyword indexes
    :return: dictionary with keywords as keys and amounts as values, amounts are seperated into each commission's year
    """
    keywords = dict()
    current_keyword = ""
    for index_link in index_links:
        response = request_website(index_link)
        soup = BeautifulSoup(response, features="html.parser")
        current_year = find_year_in_string(soup.text)
        if current_year is None:
            current_year = "unavailable"
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
    pd.options.plotting.backend = "plotly"
    matplotlib.style.use('tableau-colorblind10')
    top = dict(sorted(keyword_dict.items(), key=lambda x: sum(x[1].values()), reverse=True)[:top_n])
    df = pd.DataFrame(top).transpose()
    df = df.iloc[:, ::-1]
    df.sort_index(axis=0, ascending=False)
    fig = df.plot(kind='bar')#, stacked=True)
    fig.update_layout(
        title="Top {} keywords".format(top_n),
        xaxis_title="Keyword",
        yaxis_title="Amount mentioned",
        legend_title="Commission Years"
    )
    fig.show()
    fig.write_html("top{}keywords_plotly".format(top_n))
    #plt.xticks(rotation=45, ha='right', rotation_mode='anchor', fontsize=4)
    #plt.yticks(fontsize=7)
    # plt.yticks(new_list)
    #plt.tight_layout()
    #plt.savefig('Top{} Keywords'.format(top_n), dpi=500)
    #plt.show()


def sync_counts(k1, k2):
    for key in k2.keys():
        if key not in k1.keys():
            k1[key] = k2[key]
        else:
            k1[key] += k2[key]
    return k1


def get_overlap(s1, s2):
    """
    calculates the overlap of two strings on a character level, ignores dashes and spaces
    :param s1: first string
    :param s2: second string
    :return: the overlapping
    """
    string_list_1 = s1.replace('-', ' ').split(' ')
    string_list_2 = s2.replace('-', ' ').split(' ')
    return list(set(string_list_1).intersection(string_list_2))


def group_keys(keyword_dict):
    """
    this method groups the keys of a given dictionary by their similarity measured with the levenshtein ratio
    since this method uses simple lists, the performance is not the greatest
    :param keyword_dict: given dictionary
    :return: returns a list of lists, the latter contain a group of keywords
    """
    import time
    groups = list()
    for first_key in keyword_dict.keys():
        for group in groups:
            if all(levenshtein_distance(first_key, w) > 0.8 for w in group):
                group.append(first_key)
                break
        else:
            groups.append([first_key, ])
    return groups


def process_duplicates(keyword_dict):
    keys_to_delete = []
    for i, key in enumerate(keyword_dict.keys()):
        for j, other_key in enumerate(keyword_dict.keys()):
            dist = 0  # levenshtein_ratio_and_distance(key, other_key, ratio_calc=True)
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


def calculate_grouped_amounts(keyword_dict, grouped_keywords):
    group_dict = dict()
    for group in grouped_keywords:
        for keyword in group:
            amounts = keyword_dict[keyword]
            if str(group) not in group_dict.keys():
                group_dict[str(group)] = amounts
            else:
                for year in amounts.keys():
                    if year not in group_dict[str(group)].keys():
                        group_dict[str(group)][year] = amounts[year]
                    else:
                        group_dict[str(group)][year] += amounts[year]
    return group_dict


def main():
    commission_links = get_commission_links()
    index_links = get_index_links_from_commission_links(commission_links)
    keyword_dict = get_keywords_and_counts_basic(index_links)
    grouped_keywords = group_keys(keyword_dict)
    grouped_keyword_dict = calculate_grouped_amounts(keyword_dict, grouped_keywords)
    print("Keys saved by grouping: ", len(keyword_dict.keys()) - len(grouped_keyword_dict.keys()))
    plot_keywords(grouped_keyword_dict, 30)


if __name__ == '__main__':
    main()
