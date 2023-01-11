import math
import re

import matplotlib.style
import pandas as pd
from bs4 import BeautifulSoup
import httplib2
import matplotlib.pyplot as plt
import seaborn as sns


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
        #test = match.group(1)
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
    df_test = pd.DataFrame(top).transpose()
    df_test = df_test.iloc[:, ::-1]
    df_test.sort_index(axis=0, ascending=False)
    df_test.plot(kind='bar', stacked=True)
    plt.xticks(rotation=45, ha='right', rotation_mode='anchor', fontsize=4)
    plt.yticks(fontsize=7)
    #plt.yticks(new_list)
    plt.tight_layout()
    plt.savefig('Top{} Keywords'.format(top_n), dpi=500)
    plt.show()


def main():
    commission_links = get_commission_links()
    index_links = get_index_links_from_commission_links(commission_links)
    keyword_dict = get_keywords_and_counts_basic(index_links)
    plot_keywords(keyword_dict, 80)


if __name__ == '__main__':
    main()
