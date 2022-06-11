import os
import shutil
import re
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
from bs4 import BeautifulSoup


def get_season(season, league, rundle, division):
    matchday = 1
    div_string = f'_Div_{division}' if division else ''
    url = f'https://www.learnedleague.com/match.php?{season}&{matchday}&{rundle}_{league}{div_string}'
    fn = f'/Users/cda0201/personal/learnedleague/url.txt'
    urllib.request.urlretrieve(url, fn)

    file = open(fn, "r")
    contents = file.read()
    soup = BeautifulSoup(contents, 'html.parser')
    for data in soup.find_all("p"):
        sum = data.get_text()
        print(sum)

def parse_match(match):
    data = match.split('\t')
    pattern = re.compile('([^\(])\(([^\)])\)')
    scores = pattern.findall(data[2])

    a_mp = -1 if scores[0][1] is 'F' else int(scores[0][0])
    b_mp = -1 if scores[1][1] is 'F' else int(scores[1][0])
    a_pt = -1 if scores[0][1] is 'F' else 0
    b_pt = -1 if scores[1][1] is 'F' else 0
    a_ca = -1 if scores[0][1] is 'F' else int(scores[0][1])
    b_ca = -1 if scores[1][1] is 'F' else int(scores[1][1])
    if a_mp > b_mp:
        a_pt = 2
    elif b_mp > a_mp:
        b_pt = 2
    elif scores[0][1] is not 'F':
        a_pt = 1
        b_pt = 1
    match = ((data[1], a_ca, a_mp, a_pt),
             (data[3], b_ca, b_mp, b_pt))
    return match


def parse_matchday(matchday):
    matches = list(filter(None, matchday.split('\n')))
    matches = [parse_match(match) for match in matches]
    return matches


def parse_data(data_fn):
    with open(data_fn, 'r') as h:
        txt = h.read()
    matchdays = list(filter(None, txt.split('\n\n')))
    matchdays = [parse_matchday(matchday) for matchday in matchdays]
    return matchdays


def luck(data_fn, player='DunnC4'):
    matchdays = parse_data(data_fn)

    n_matchdays = len(matchdays)
    n_players = len(matchdays[0])*2

    cas = np.zeros((n_matchdays, n_players - 1), dtype=int)
    mps = np.zeros((n_matchdays, n_players - 1), dtype=int)
    pts = np.zeros((n_matchdays, n_players - 1), dtype=int)
    player_cas = np.zeros((n_matchdays, ), dtype=int)
    player_cps = np.zeros((n_matchdays, ), dtype=int)
    player_mps = np.zeros((n_matchdays, ), dtype=int)
    player_pts = np.zeros((n_matchdays, ), dtype=int)
    opp_cas = np.zeros((n_matchdays, ), dtype=int)
    opp_mps = np.zeros((n_matchdays, ), dtype=int)
    opp_pts = np.zeros((n_matchdays, ), dtype=int)
    for d, matchday in enumerate(matchdays):
        ind = 0
        for match in matchday:
            for r, result in enumerate(match):
                if result[0] == player:
                    player_cas[d] = result[1]
                    player_mps[d] = result[2]
                    player_pts[d] = result[3]
                    opp_cas[d] = match[np.mod(r + 1, 2)][1]
                    opp_mps[d] = match[np.mod(r + 1, 2)][2]
                    opp_pts[d] = match[np.mod(r + 1, 2)][3]
                    player_cps[d] = 2*(player_cas[d] > opp_cas[d]) + (player_cas[d] == opp_cas[d])*(player_cas[d] > -1) - (player_cas[d] == -1)
                else:
                    cas[d, ind] = result[1]
                    mps[d, ind] = result[2]
                    pts[d, ind] = result[3]
                    ind += 1

    cads = player_cas[:, np.newaxis] - cas
    mpds = player_mps[:, np.newaxis] - mps
    ptds = player_pts[:, np.newaxis] - pts
    opp_cads = player_cas - opp_cas
    opp_mpds = player_mps - opp_mps
    opp_ptds = player_pts - opp_pts

    ps = np.tile(player_cas[:, np.newaxis], (1, n_players - 1))
    os = cas
    cad_points = 2*(ps > os) + (ps == os)*(ps > -1) - (ps < 0)

    ps = np.tile(player_mps[:, np.newaxis], (1, n_players - 1))
    os = mps
    mpd_points = 2*(ps > os) + (ps == os)*(ps > -1) - (ps < 0)


    fg, axs = plt.subplots(ncols=4, nrows=n_matchdays, figsize=(12, 8))
    sources = [cads, cad_points, mpds, mpd_points]
    opp_sources = [opp_cads, player_cps, opp_mpds, player_pts]
    source_labels = ['Correct Answer Difference (opp <-> me)',
                     'Points based on Correct Answers',
                     'Match Point Difference (opp <-> me)',
                     'Points']
    source_ranges = [[-7, 7], [-1, 2], [-10, 10], [-1, 2]]
    source_colors = ['k', 'dimgrey', 'b', 'g']

    ofg, oaxs = plt.subplots(nrows=4, figsize=(8, 16))
    for s in range(len(sources)):
        max_count = -np.inf
        bin_edges = np.arange(source_ranges[s][0], source_ranges[s][1] + 2) - .5
        counts = np.zeros((n_matchdays, len(bin_edges) - 1), dtype=int)
        ds = sources[s]
        overall_d = np.ones((1, ))
        for d, ax in enumerate(axs[:, s]):
            counts[d, :], bins, patches = ax.hist(ds[d, :], bins=bin_edges, edgecolor='white', linewidth=1)

            for i in range(0, opp_sources[s][d] - source_ranges[s][0]):
                patches[i].set_facecolor(source_colors[s])
            patches[opp_sources[s][d] - source_ranges[s][0]].set_facecolor('r')
            for i in range(opp_sources[s][d] - source_ranges[s][0] + 1, len(patches)):
                patches[i].set_facecolor(source_colors[s])

            ax.set_xlim([source_ranges[s][0] - 1, source_ranges[s][1] + 1])
            if d == n_matchdays - 1:
                if s == 0 or s == 2:
                    ax.set_xticks(np.arange(source_ranges[s][0] + 1, source_ranges[s][1])[::2])
                else:
                    ax.set_xticks(np.arange(source_ranges[s][0], source_ranges[s][1] + 1))
                ax.set_xlabel(source_labels[s])
            else:
                ax.set_xticks([0, ])
                ax.tick_params(
                    axis='x',  # changes apply to the x-axis
                    which='both',  # both major and minor ticks are affected
                    labelbottom=False)  # labels along the bottom edge are off
            if s == 1 and d == 0:
                ax.set_title(player)
            if s == 0:
                ax.set_ylabel(f'MD {d+1}')
            max_count = np.maximum(np.max(counts[d, :]), max_count)
            overall_d = np.convolve(overall_d, counts[d, :])

        bext = [source_ranges[s][0] * n_matchdays, source_ranges[s][1] * n_matchdays]
        _, _, patches = oaxs[s].hist(np.arange(bext[0], bext[1] + 1), weights=overall_d,
                                     bins=np.arange(bext[0], bext[1] + 2) - .5, edgecolor='white', linewidth=1, density=True)

        red_val = np.sum(opp_sources[s])
        red_ind = np.argmax(np.arange(bext[0], bext[1] + 1) == red_val)
        for i in range(0, red_ind):
            patches[i].set_facecolor(source_colors[s])
        patches[red_ind].set_facecolor('r')
        for i in range(red_ind + 1, len(patches)):
            patches[i].set_facecolor(source_colors[s])

        if s == 0 or s == 2:
            trim = np.minimum(np.argmax(overall_d > 0), np.argmax(overall_d[::-1] > 0))
            bext = [bext[0] + trim, bext[1] - trim]
            oaxs[s].set_xlim([bext[0] - 1, bext[1] + 1])
        else:
            oaxs[s].set_xlim([bext[0] - 1, bext[1] + 1])

        oaxs[s].set_xlabel(source_labels[s])
        if s == 0:
            oaxs[s].set_title(player)
        # ofg.savefig(f'/Users/cda0201/personal/learnedleague/{player}_overall_{s}_{d}.png')

        for ax in axs[:, s]:
            ax.set_ylim([0, max_count + 1])

    ofg.savefig(f'/Users/cda0201/personal/learnedleague/{player}_overall.png')

    fg.savefig(f'/Users/cda0201/personal/learnedleague/{player}.png')

    return

if __name__ == '__main__':

    season = 92
    rundle = 'D'
    division = 1
    league = 'Citadel'
    get_season(season, league, rundle, division)

    season = 92
    rundle = 'D'
    data_fn = f'/Users/cda0201/personal/learnedleague/LL{season}_{rundle}'
    players = ['DunnC4', 'CampbellJC', 'LouJ', 'Watson JrJ']
    for player in players:
        luck(data_fn, player)

    season = 92
    rundle = 'B'
    data_fn = f'/Users/cda0201/personal/learnedleague/LL{season}_{rundle}'
    players = ['DunnGreg']
    for player in players:
        luck(data_fn, player)

    season = 92
    rundle = 'C2'
    data_fn = f'/Users/cda0201/personal/learnedleague/LL{season}_{rundle}'
    players = ['WatsonJ3', 'WatsonL3']
    for player in players:
        luck(data_fn, player)

    # season = 86
    # rundle = 'R'
    # data_fn = f'/Users/cda0201/personal/learnedleague/LL{season}_{rundle}'
    # players = ['DunnC4']
    # for player in players:
    #     luck(data_fn, player)