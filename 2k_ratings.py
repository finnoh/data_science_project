import requests
from bs4 import BeautifulSoup
import time
from tqdm import tqdm
import pandas as pd

def get_2k_attributes(url):
    r = requests.get(url, timeout=2.5)
    r_html = r.text
    soup = BeautifulSoup(r_html, 'html.parser')

    names_raw = soup.find_all('a', class_ = 'name box-link')
    names = [n.text.strip() for n in names_raw]
    scores_raw = soup.find_all('td', class_ = 'attribute')

    scores = []
    temp = []
    for s in scores_raw:
        score = " ".join(s.text.split())
        if score == '-':
            scores.append(temp)
            temp = []

        else:
            temp.append(int(score))

    if len(temp) != 0:
        scores.append(temp)

    return dict(zip(names, scores))

ratings = get_2k_attributes('https://2kmtcentral.com/21/players/collection/325-326-327-328-329-330-331-332-333-334-335-336-337-338-339-340-341-342-343-344-345-346-347-348-349-350-351-352-353-354')

for page in tqdm(range(1, 15)):
    time.sleep(5)
    url = f'https://2kmtcentral.com/21/players/collection/325-326-327-328-329-330-331-332-333-334-335-336-337-338-339-340-341-342-343-344-345-346-347-348-349-350-351-352-353-354/page/{page}'
    ratings.update(get_2k_attributes(url))

col_names = []
col_total = []
col_inside = []
col_outside = []
col_playmaking = []
col_athleticism = []
col_defending = []
col_rebounding = []

for p,r in ratings.items():
    col_names.append(p)
    col_total.append(r[0])
    col_inside.append(r[1])
    col_outside.append(r[2])
    col_playmaking.append(r[3])
    col_athleticism.append(r[4])
    col_defending.append(r[5])
    col_rebounding.append(r[6])

ratings_df = pd.DataFrame()
ratings_df['player'] = col_names
ratings_df['total'] = col_total
ratings_df['inside'] = col_inside
ratings_df['outside'] = col_outside
ratings_df['playmaking'] = col_playmaking
ratings_df['athleticism'] = col_athleticism
ratings_df['defending'] = col_defending
ratings_df['rebounding'] = col_rebounding

ratings_df.to_csv('data/rec_engine/nba2k_ratings.csv', index = False)