# HMM Irregularity 

Continuous Hidden Time Markov Chain for tracking the evolution of irregularity, namely Paul's principle, in the evolution of High German's strong verbs. The principle states that verbs that have high degree of irregularity (vowel and consonant alternations in stem) are less likely to regularize than verbs that have only some degree of irregularity (vowel or consonant alternation in the stem).

The dependencies required to run the code are listed in `environment.yml`. You can create a new conda environment directly from it using `conda env create --file environment.yml`. 

**NOTE:** There are some differences in installation of Stan for different platforms using conda. If you use `environment.yml`, Windows environment and certain distributions of Linux (apart from Debian, see below) should work out of the box, however, you may see some additional warnings that are safe to ignore, if the model starts to run. For more details, take a look at the [official Stan page](https://mc-stan.org/install/). To summarize the installation via conda: 

- Mac users additionally have to install C++ compiler. The recommended option for it is to run `xcode-select --install`
- In case of Debian, you also have to install C++ compiler. The option from the official installation guide is to run `sudo apt-get install build-essential`

