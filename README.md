# Reconstructing Punctuation

Final project for the machine learning lecture in Heidelberg, winter term 2015/2016.
By Martin Bidlingmaier, Albert Marciniak and Christian Straßberger.


## Setup

Assuming you're on linux machine:
Make sure you have installed

* python3
* pip
* virtualenv
* a fortran compiler (e.g. gcc-fortran).
* svn
* basic unix build utilities like make and a c compiler

Open a terminal in the project root directory and run
```shell
virtualenv env
source env/bin/activate
pip install $(cat requirements.txt)
./setup-tools
```
If you want to use matplotlib, make sure you have installed matplotlib system wide (e.g. `pacman -S python-matplotlib` if you're on arch linux) and replace `virtualenv env` by `virtualenv --system-site-packages env`.

To download and setup basic data, run `./setup_data.py`.
The download may take *very* long (multiple days).
