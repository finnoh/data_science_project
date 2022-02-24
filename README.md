# Data Science Project: NBA Dashboard

This repository hosts the Data Science Project (DS500) of Finn Höner and Tim-Moritz Bündert.
It contains all the commented code associated with the project along with the corresponding data.

The dashboard (using the **app.py** file) is hosted on [this website](http://193.196.53.114/).

Larger data files can be accessed on [Google Drive](https://drive.google.com/drive/folders/1nl-O5oP8OEU6t839dVkCwwlSNc3XAY1e?usp=sharing).

Should you be interested in more information regarding this project, feel free to contact us.

finn.hoener@student.uni-tuebingen.de
tim-moritz.buendert@student.uni-tuebingen.de

## Structure

Generally, all files which are not factored into any subfolders are required for the successful deployment of the dashboard app. When cloning the repository, no file paths need to be changed to run the project, only the *Model* folder from the Google Drive link below needs to be added (neglected here, because of its size).

Different subfolders and their purposes:

### data
Contains datafiles, optionally odered in more subfolders for different applications. We should pay attention, if there is a potential to make use of multiple dataset jointly.

### assets
Later on this can be used for style elements such as `.css` files or images.

### notes
Folder containing notes in `.md` format and their generated PDFs.

### backlog
Notebooks and further files which were used to construct intermediate results, but not relevant for production.

### src
Folder for source code, i.e. here we can specify python modules

# Disclaimer
The corresponding data along with the names and images displayed in the dashboard are not our property. Rather, the following references apply:
* [NBA](https://www.nba.com/termsofuse), in particular 
    - Images of teams: http://i.cdn.turner.com/nba/nba/.element/img/1.0/teamsites/logos/teamlogos_500x500/....png
    - Images of players: https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/....png
* [NBA_API](https://github.com/swar/nba_api)
* Team salaries from https://www.spotrac.com/nba/cap/
* Player salaries from https://hoopshype.com/salaries/players/
* [NBA2k ratings](https://2kmtcentral.com/21/players/collection/)
* [RAPM Scores](http://nbashotcharts.com/home)
* [NBA Logo](https://logosmarken.com/nba-logo/)
* [FiveThirtyEight](https://projects.fivethirtyeight.com/2022-nba-predictions/raptors/)
