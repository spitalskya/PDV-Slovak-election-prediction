# Slovak election prediction

*contributors: Tomáš Antal, Erik Božík, Teo Pazera, Andrej Špitalský, Tomáš Varga*

In this repository, you can find materials used to analyze, classify and predict election results in Slovak republic based on election polls, socio-economic data and political stances of individual parties.

Report of our work can be found in `tex/report.pdf`.

You can run our model to predict elections in Slovakia (if they were held in December of 2024 and in May 2025 by):

```
make future_elections
```

or by 

```
python -m src.predict_future_elections
```

Other parts of the project can be run in similar fashion, either through `Makefile` rules or by `python -m` command. Jupyter notebooks can be run individually in their own environment.

This repository was created as a project for an university course *1-DAV-302/20 -- Princípy dátovej vedy* by Mgr. Samuel Rosa, PhD. in Faculty of Mathematics, Physics and Informatics, Comenius University, Bratislava, Slovakia.