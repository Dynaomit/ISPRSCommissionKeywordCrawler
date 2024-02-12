import pickle
import re
from itertools import combinations

from tqdm import tqdm
from fuzzywuzzy import fuzz

import matplotlib.style
import pandas as pd
from bs4 import BeautifulSoup
import httplib2
from collections import Counter
import plotly.graph_objects as go


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
    for link in tqdm(BeautifulSoup(response, features="html.parser").find_all('a'),
                     desc="Parsing Website for Commissions"):
        link_ref = link.get('href')
        link_name = link.text
        if link_ref is not None \
                and link_name is not None \
                and "isprs-annals.copernicus.org" in link_ref \
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
    for commission_link in tqdm(commission_links, desc="Processing Commissions"):
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


keyword_mapping = {}


def get_umbrella_term(keyword):
    global keyword_mapping

    if keyword in keyword_mapping:
        return keyword_mapping[keyword]

    for existing_keyword in keyword_mapping:
        if fuzz.ratio(keyword, existing_keyword) >= 80:
            return keyword_mapping[existing_keyword]

    keyword_mapping[keyword] = keyword

    return keyword


def get_keywords_and_counts_basic(index_links):
    """
    from a given list of links referring to keyword indexes,
    parses the keywords and counts how many papers mention each keyword
    the keywords are collected in a dictionary as keys to prevent duplicates
    :param index_links: given list of links to keyword indexes
    :return: dictionary with keywords as keys and amounts as values, amounts are seperated into each commission's year
    """
    keywords = dict()
    papers = dict()
    current_keyword = ""
    for index_link in tqdm(index_links, desc="Processing Keyword Indices"):
        response = request_website(index_link)
        soup = BeautifulSoup(response, features="html.parser")
        current_year = find_year_in_string(soup.text)
        if current_year is None:
            current_year = "unavailable"
        mydivs = soup.find_all("div", {"class": "indexData"})
        for div in mydivs:
            for content in div.contents:
                if content.name == "span":
                    current_keyword = content.get_text(strip=True)
                if content.name == "ul":
                    paper_split = content.get_text(strip=False).split('\n')
                    paper_split = [paper for paper in paper_split if paper.strip() != ""]
                    for paper in paper_split:
                        keyword_to_add = current_keyword
                        if paper not in papers:
                            papers[paper] = [keyword_to_add]
                        if keyword_to_add not in papers[paper]:
                            papers[paper].append(keyword_to_add)
                        if current_keyword not in keywords:
                            keywords[current_keyword] = {current_year: 1}
                        else:
                            if current_year not in keywords[current_keyword]:
                                keywords[current_keyword][current_year] = 1
                            else:
                                keywords[current_keyword][current_year] += 1
    all_keywords = list(set(k for ks in papers.values() for k in ks))

    threshold = 3

    cooccurrences = Counter()
    for ks in papers.values():
        keyword_combinations = combinations(ks, 2)
        cooccurrences.update(keyword_combinations)

    frequent_pairs = {pair: count for pair, count in cooccurrences.items() if count >= threshold}

    sorted_pairs = sorted(frequent_pairs.items(), key=lambda x: x[1], reverse=True)

    return keywords, sorted_pairs


def plot_keywords(keyword_dict, top_n):
    """
    creates a bar graph that visualizes the number of mentions for a given dict of keywords
    can be limited to only show the top n keywords by overall mentions
    :param keyword_dict:
    :param top_n:
    """
    pd.options.plotting.backend = "plotly"
    matplotlib.style.use('tableau-colorblind10')
    top = dict(sorted(keyword_dict.items(), key=lambda x: sum(x[1].values()), reverse=True)[:top_n])
    #top = dict(sorted(keyword_dict.items(), key=lambda x: x[1].get('2022', 0), reverse=True)[:top_n])
    with open(r'./keywords.txt', 'w') as fp:
        for key in top.keys():
            fp.write("%s\n" % (key[0].upper() + key[1:]))
    for key in list(top.keys()):
        new_key = key.strip('][').split(', ')[0].strip('\'')
        new_key = new_key[0].upper() + new_key[1:]
        top[new_key] = top[key]
        del top[key]
    df = pd.DataFrame(top).transpose()
    df = df.iloc[:, ::-1]
    df.sort_index(axis=0, ascending=False)
    fig = df.plot(kind='bar')
    fig.update_layout(
        title="Top {} keywords".format(top_n),
        xaxis_title="Keyword",
        yaxis_title="Amount mentioned",
        legend_title="Congress Years",
        title_font=dict(size=40),
        xaxis=dict(
            tickfont=dict(size=20),
            title_font=dict(size=20)
        ),
        yaxis=dict(
            tickfont=dict(size=20),
            title_font=dict(size=20)
        ),
        legend=dict(font=dict(size=20))
    )
    fig.show()
    fig.write_html("top{}keywords_plotly.html".format(top_n))


def group_keys(keyword_dict):
    """
    this method groups the keys of a given dictionary by their similarity measured with the levenshtein ratio
    since this method uses simple lists, the performance is not the greatest
    :param keyword_dict: given dictionary
    :return: returns a list of lists, the latter contain a group of keywords
    """
    groups = list()
    for first_key in keyword_dict.keys():
        for group in groups:
            #if all(fuzz.ratio(first_key, w) >= 80 for w in group):
            #    group.append(first_key)
            #    break
            if fuzz.ratio(first_key, group[0]) >= 80:
                group.append(first_key)
                break
        else:
            groups.append([first_key, ])
    return groups


def calculate_grouped_amounts(keyword_dict, grouped_keywords):
    """
    summarizes all values in a given dict for all keywords in a group
    :param keyword_dict: input dict
    :param grouped_keywords: keyword groups
    :return: grouped and summarizes dict
    """
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


def process_pair_keywords(keywords, pair_frequencies):
    """
    creates a dict of keyword pairs, their frequencies together and alone
    :param keywords:
    :param pair_frequencies:
    :return: resulting list of keyword pairs
    """
    sum1 = 0
    sum2 = 0
    for i, (pair, frequency) in enumerate(pair_frequencies):
        keyword1, keyword2 = pair
        for key in keywords.keys():
            keyword_group = eval(key)
            if keyword1 in keyword_group:
                keyword1 = keyword_group[0]
                sum1 = sum(keywords[key].values())
            if keyword2 in keyword_group:
                keyword2 = keyword_group[0]
                sum2 = sum(keywords[key].values())
        pair_frequencies[i] = ((keyword1, keyword2), frequency, (sum1, sum2))

    duplicate_free_pairs = {}

    for pair, frequency, sums in pair_frequencies:
        if pair in duplicate_free_pairs:
            duplicate_free_pairs[pair][0] += frequency
            assert(duplicate_free_pairs[pair][1] == sums)
        else:
            duplicate_free_pairs[pair] = [frequency, sums]

    duplicate_free_pairs = [(pair, frequency, sums) for pair, (frequency, sums) in duplicate_free_pairs.items()]
    duplicate_free_pairs.sort(key=lambda x: x[1], reverse=True)
    return duplicate_free_pairs


def plot_map(pair_frequencies):
    """
    creates a force-directed graph
    :param pair_frequencies: list that contains keyword pairs and their frequencies
    """
    with open("pair_frequencies.pkl", "wb") as f:
        pickle.dump(pair_frequencies, f)
    import networkx as nx

    graph = nx.Graph()
    for (keyword1, keyword2), frequency, (sum1, sum2) in pair_frequencies:
        k1 = keyword1[0].upper() + keyword1[1:]
        k2 = keyword2[0].upper() + keyword2[1:]
        graph.add_node(k1, weight=sum1*0.5)
        graph.add_node(k2, weight=sum2*0.5)
        graph.add_edge(k1, k2, frequency=frequency)

    layout = nx.spring_layout(graph)

    node_positions = {node: (pos[0], pos[1]) for node, pos in layout.items()}

    edge_traces = []
    for edge in graph.edges():
        x0, y0 = node_positions[edge[0]]
        x1, y1 = node_positions[edge[1]]
        frequency = graph.edges[edge]["frequency"]
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode="lines",
            line=dict(width=frequency, color="gray"),
            hoverinfo="none",
        )
        edge_traces.append(edge_trace)

    # Define node traces
    node_traces = []
    for node in graph.nodes():
        x, y = node_positions[node]
        frequency = graph.nodes[node]["weight"]*2
        node_trace = go.Scatter(
            x=[x],
            y=[y],
            mode="markers",
            marker=dict(size=frequency, color="skyblue", line=dict(color='black', width=1)),
            text=f"{node} ({frequency})",
            hoverinfo="text",
        )
        node_traces.append(node_trace)

        label_trace = go.Scatter(
            x=[x],
            y=[y],
            mode="text",
            text=[node],
            textposition="middle center",
            hoverinfo="none",
            showlegend=False,
            textfont=dict(
                color="black",
                size=12
            )
        )
        node_traces.append(label_trace)

    # Create figure
    figure = go.Figure(data=edge_traces + node_traces)

    # Update layout
    figure.update_layout(
        title="Force-Directed Graph For Keyword Relation",
        showlegend=False,
        hovermode="closest",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        title_font=dict(size=40),
    )

    # Show the graph
    figure.show()
    figure.write_html("keyword_frequency_and_relation_map.html")


def main():
    commission_links = get_commission_links()
    index_links = get_index_links_from_commission_links(commission_links)
    keyword_dict, sorted_frequency_pairs = get_keywords_and_counts_basic(index_links)
    filtered_keys = [key for key in keyword_dict.keys() if
                     '/' not in key and '\\' not in key and '<' not in key and '>' not in key]
    filtered_empty_dict = {key: {} for key in filtered_keys}
    with open('filtered_keys.pkl', 'wb') as f:
        pickle.dump(list(filtered_empty_dict), f)
    grouped_keywords = group_keys(keyword_dict)
    grouped_keyword_dict = calculate_grouped_amounts(keyword_dict, grouped_keywords)
    print("Performed grouping optimization")
    print("Keys saved by grouping: ", len(keyword_dict.keys()) - len(grouped_keyword_dict.keys()))
    plot_keywords(grouped_keyword_dict, 100)
    grouped_pair_frequencies = process_pair_keywords(grouped_keyword_dict, sorted_frequency_pairs)
    plot_map(grouped_pair_frequencies)
    plot_frequency_pairs(grouped_pair_frequencies)


def plot_frequency_pairs(sorted_frequency_pairs):
    """
    creates a graph that shows keyword pairs and their frequencies in a scatter plot
    :param sorted_frequency_pairs: keyword pairs and frequencies in a sorted structure
    """
    keywords = [f"{pair[0][0]}, {pair[0][1]}" for pair in sorted_frequency_pairs]
    frequencies = [pair[1] for pair in sorted_frequency_pairs]
    with open("keywords_for_graph.pkl", 'wb') as f:
        pickle.dump(keywords, f)
    with open("frequencies_for_graph.pkl", 'wb') as f:
        pickle.dump(frequencies, f)
    fig = go.Figure(data=go.Scatter(x=keywords, y=frequencies, mode='markers'))
    fig.update_layout(title='Keyword Pair Frequencies',
                      xaxis_title='Keyword Pair',
                      yaxis_title='Frequency',
                      title_font=dict(size=40))
    fig.show()
    fig.write_html("frequency_pairs.html")


if __name__ == '__main__':
    main()
