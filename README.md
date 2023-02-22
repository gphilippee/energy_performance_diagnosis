# RAMP starting kit on Energy Consumption (EC) and Greenhouse gases (GHG) classification

_Authors: Cyril Zaimi, Guillaume Philippe, Matteo Ratet, Max Wu, Pierre Loviton, & Hugo Peltier_

This challenge was done as a project for the Master 2 Data Science (2022/2023), DATACAMP course

# TODO
- [x] Write paragraph for introduction
- [ ] Implement metrics (weight between classes, mixed, ...)
- [x] Implement cross-validation
- [ ] Implement submission
- [ ] Encode categorical features
- [ ] Improve starting-kit

## Introduction

One of the main levers of the ecological transition is to act on emissions from buildings. Indeed, according to the International Energy Agency, they account for approximately 28% of national greenhouse gas emissions. This is why this sector is at the heart of a massive investment plan launched in 2021 by the government. France has set itself the goal of reducing the energy consumption of existing buildings by 38% by 2030. 

The **Energy Performance Diagnostic** (EPD) is a tool that has been introduced to evaluate the energy efficiency of buildings. Its primary purpose is to help identify areas where a structure is wasting energy and suggest ways to improve energy performance. This can involve recommendations on how to upgrade insulation, replace inefficient heating or cooling systems, install energy-efficient lighting, and more.

According to recent estimates, of the 30 million primary residences in metropolitan France, as of January 1st, 2022, approximately 5.2 million homes would be an energy leak (labels F and G). This project aims at constructing a model capable of estimating the **energy consumption** (EC) and **greenhouse gas emissions** (GGE) to locate more precisely the renovations to be undertaken in priority.

#### Useful links

- [Features description](https://koumoul.com/data-fair/embed/dataset/dpe-tertiaire/fields)

#### Set up

Open a terminal and

1. install the `ramp-workflow` library and the dependencies
```shell
$ pip install -r requirements.txt
```
  
2. Follow the ramp-kits instructions from the [wiki](https://github.com/paris-saclay-cds/ramp-workflow/wiki/Getting-started-with-a-ramp-kit)

#### Local notebook

Get started on this RAMP with the [dedicated notebook](solar_wind_starting_kit.ipynb).

To test the starting-kit, run

```shell
ramp-test --quick-test
```

#### Help
Go to the `ramp-workflow` [wiki](https://github.com/paris-saclay-cds/ramp-workflow/wiki) for more help on the [RAMP](https://ramp.studio) ecosystem.
